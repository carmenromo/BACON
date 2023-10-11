### BLR functions based adapted from IC

import numpy as np

from scipy import signal as sgn

coeff_c_NEXT   = np.array([0.00000042504, 0.00000055255, 0.00000035999, 0.00000035639, 0.00000047635, 0.00000059184, 0.00000055974, 0.00000057049, 0.00000038041, 0.00000041167, 0.00000037353, 0.00000048496])
coeff_blr_NEXT = np.array([0.00053015, 0.00052996, 0.00053319, 0.00052624, 0.00052731, 0.00052813, 0.00051798, 0.00052867, 0.00052046, 0.0005271, 0.00052704, 0.00052508])


def deconvolve_signal(signal_daq,
                      coeff_clean            = 2.905447E-06,
                      coeff_blr              = 1.632411E-03,
                      baseline_mode          = False,
                      wf_range_bsl           = (0, None),
                      thr_trigger            =     5,
                      accum_discharge_length =  5000):
    """
    Function adapted from IC
    """
    
    thr_acum = thr_trigger / coeff_blr
    len_signal_daq = len(signal_daq)

    # Compute baseline
    if baseline_mode:
        baseline = st.mode(signal_daq[wf_range_bsl[0]:wf_range_bsl[1]], keepdims=False).mode.astype(np.float32)
    else:
        baseline = np.mean(signal_daq[wf_range_bsl[0]:wf_range_bsl[1]])

    # Reverse sign of signal and subtract baseline
    signal_daq_bs = baseline - signal_daq

    nn = 400  # fixed at 10 mus
    # Compute noise
    noise = np.sum(signal_daq_bs[:nn] ** 2) / nn
    noise_rms = np.sqrt(noise)

    # trigger line
    trigger_line = thr_trigger * noise_rms

    b_cf, a_cf    = sgn.butter(1, coeff_clean, 'high', analog=False)
    signal_daq_bs = sgn.lfilter(b_cf, a_cf, signal_daq_bs)

    signal_r = np.empty(len_signal_daq)
    acum     = np.zeros(len_signal_daq, dtype=np.double)
    
    # Always update signal and accumulator
    signal_r[0]    = signal_daq_bs[0]
    coeff_blr_half = coeff_blr / 2
    
    trigger_mask   = (signal_daq_bs[:-1] < trigger_line) & (acum[:-1] < thr_acum)
    acum_discharge = np.where(trigger_mask, np.maximum(1, acum[:-1]) * (1 - coeff_blr), 0)
    
    j_values = np.arange(accum_discharge_length)
    j_values = j_values[j_values < np.sum(trigger_mask)]
    acum[1:] = np.where(trigger_mask, acum[:-1] + signal_daq_bs[1:], acum[1:])
    
    # Discharge the accumulator
    acum[1:] -= acum_discharge
    
    signal_r[1:] = signal_daq_bs[1:] + signal_daq_bs[1:] * coeff_blr_half + coeff_blr * acum[1:]

    return np.asarray(signal_r)


def blr_deconv_pmt(pmtrwf,
                   coeff_c,
                   coeff_blr,
                   baseline_mode          = False,
                   wf_range_bsl           = (0, None),
                   thr_trigger            =    5,
                   accum_discharge_length = 5000):
    """
    Deconvolve all the PMTs in the event.
    :param pmtrwf: array of PMTs holding the raw waveform
    :param coeff_c:     cleaning coefficient
    :param coeff_blr:   deconvolution coefficient
    :param n_baseline:  number of samples taken to compute baseline
    :param thr_trigger: threshold to start the BLR process
    
    :returns: an array with deconvoluted PMTs.
    """

    nwf      = len(pmtrwf)
    signal_i = pmtrwf.astype(np.double)
    signal_r = np.zeros(nwf, dtype=np.double)

    signal_r = deconvolve_signal(signal_i,
                                 coeff_clean            = coeff_c,
                                 coeff_blr              = coeff_blr,
                                 baseline_mode          = baseline_mode,
                                 wf_range_bsl           = wf_range_bsl,
                                 thr_trigger            = thr_trigger,
                                 accum_discharge_length = accum_discharge_length)
    return signal_r


def pmt_deconvolver(rwf,
                    baseline_mode = False,
                    wf_range_bsl  = (0, None),):
    coeff_c    = np.mean(coeff_c_NEXT)
    coeff_blr  = np.mean(coeff_blr_NEXT)

    return blr_deconv_pmt(rwf,
                          coeff_c,
                          coeff_blr,
                          baseline_mode=baseline_mode,
                          wf_range_bsl=wf_range_bsl)