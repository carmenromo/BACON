### Fit functions from IC
import math
import scipy
import numpy as np

import matplotlib.pyplot as plt
from collections import namedtuple

FitFunction = namedtuple('FitFunction', ['fn', 'values', 'errors', 'chi2', 'pvalue', 'cov'])


def in_range(data, minval=-np.inf, maxval=np.inf, left_closed=True, right_closed=False):
    lower_bound = data >= minval if left_closed  else data > minval
    upper_bound = data <= maxval if right_closed else data < maxval
    return lower_bound & upper_bound

def get_errors(cov):
    return np.sqrt(np.diag(cov))

def poisson_sigma(x, default=3):
    u = x**0.5
    u[x==0] = default
    return u

def get_chi2_and_pvalue(ydata, yfit, ndf, sigma=None):
    if sigma is None:
        sigma = poisson_sigma(ydata)

    chi2   = np.sum(((ydata - yfit) / sigma)**2)
    pvalue = scipy.stats.chi2.sf(chi2, ndf)

    return chi2 / ndf, pvalue

def gauss(x, area, mu, sigma):
    if sigma <= 0.:
        return np.inf
    return area/(2*np.pi)**.5/sigma * np.exp(-0.5*(x-mu)**2./sigma**2.)

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

def multi_gaussian(x, *params):
    n_peaks = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n_peaks):
        amplitude = params[i * 3]
        mean = params[i * 3 + 1]
        stddev = params[i * 3 + 2]
        y += gaussian(x, amplitude, mean, stddev)
    return y

def fit(func, x, y, seed=(), fit_range=None, **kwargs):
    if fit_range is not None:
        sel  = in_range(x, *fit_range)
        x, y = x[sel], y[sel]
        if "sigma" in kwargs:
            kwargs["sigma"] = kwargs["sigma"][sel]

    sigma_r = kwargs.get("sigma", np.ones_like(y))
    if np.any(sigma_r <= 0):
        raise ValueError("Zero or negative value found in argument sigma. "
                         "Errors must be greater than 0.")

    kwargs['absolute_sigma'] = "sigma" in kwargs

    vals, cov = scipy.optimize.curve_fit(func, x, y, seed, **kwargs)

    fitf       = lambda x: func(x, *vals)
    fitx       = fitf(x)
    errors     = get_errors(cov)
    ndof       = len(y) - len(vals)
    chi2, pval = get_chi2_and_pvalue(y, fitx, ndof, sigma_r)

    return FitFunction(fitf, vals, errors, chi2, pval, cov)

def shift_to_bin_centers(x):
    return x[:-1] + np.diff(x) * 0.5

def f_values(f):
    _, mu,     sigma     = f.values
    _, mu_err, sigma_err = f.errors
    fwhm, fwhm_err = sigma* 2.355, sigma_err* 2.355
    return mu, mu_err, sigma, sigma_err, fwhm, fwhm_err, f.chi2

def string_fit(f, units='ADC', ndec=2, print_chi2=True):
    mu, mu_err, sigma, sigma_err, fwhm, fwhm_err, chi2 = f_values(f)
    mu_str   = f'$\mu$ = {np.round(mu, ndec)} ± {np.round(mu_err, ndec)} {units}'
    sig_str  = f'$\sigma$ = {np.round(sigma, ndec)} ± {np.round(sigma_err, ndec)} {units}'
    chi2_str = f'$\chi^2$ / ndf = {np.round(chi2,  ndec)}'
    fwhm_str = f'FWHM = {np.round(fwhm, 2)} ± {np.round(fwhm_err, 2)} pes'
    if print_chi2:
        return f'{mu_str}'+'\n'+f'{sig_str}'+'\n'+f'{chi2_str}'
    else:
        return f'{mu_str}'+'\n'+f'{sig_str}'
    
def gaussian_fit_IC(data, bins=100, prange=(-50, 150), ampl=100, mean=0, sigma=10, frange=(-50, 50), title='', xlabel='Amplitude (ADC)', units='ADC', ndec=2, print_chi2=True, figsize=(8,5)):
    plt.figure(figsize=figsize)
    y, x, _ = plt.hist(data, bins=bins, alpha=0.5, range=prange, histtype='step', linewidth=1.5)
    plt.axvspan(frange[0], frange[1], color='grey', alpha=0.2)
    f = fit(gauss, shift_to_bin_centers(x), y, (ampl,mean,sigma), fit_range=frange, sigma=np.sqrt(y))

    plt.plot(shift_to_bin_centers(x), gauss(shift_to_bin_centers(x), *f.values[:3]), 'r--',
             label=string_fit(f, ndec=ndec, print_chi2=print_chi2, units=units))
    plt.errorbar(shift_to_bin_centers(x), y, yerr=np.sqrt(y), fmt='.k', elinewidth=0.5, capsize=2, capthick=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Entries/bin')
    plt.legend()
    plt.show()
    
    return f_values(f)

def gaussian_fit_IC_subplot(ax, data, bins=100, prange=(-50, 150), ampl=100, mean=0, sigma=10, frange=(-50, 50), title='', xlabel='Amplitude (ADC)', units='ADC', ndec=2, print_chi2=True):
    y, x, _ = ax.hist(data, bins=bins, alpha=0.5, range=prange, histtype='step', linewidth=1.5)
    f = fit(gauss, shift_to_bin_centers(x), y, (ampl,mean,sigma), fit_range=frange, sigma=np.sqrt(y))

    ax.plot(shift_to_bin_centers(x), gauss(shift_to_bin_centers(x), *f.values[:3]), 'r--',
             label=string_fit(f, ndec=ndec, print_chi2=print_chi2, units=units))
    ax.errorbar(shift_to_bin_centers(x), y, yerr=np.sqrt(y), fmt='.k', elinewidth=0.5, capsize=2, capthick=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Entries/bin')
    ax.legend(loc='upper right', handlelength=0.5)
    
    return f_values(f)

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def plot_linear_fit(y, yerr):
    x    = np.arange(len(y))+1
    slope, intercept, r_value, _, _ = scipy.stats.linregress(x, y)
    line = slope*x + intercept

    plt.figure(figsize=(8,5))
    plt.errorbar(x, y, yerr=yerr, marker='_', markersize=5, linestyle='', c='k', label=f'Measured gain values')
    plt.plot(x, line, color='r', alpha=0.7, label=f'Fit: y = x*{round(slope, 2)} - {round(np.abs(intercept), 2)}, \n     R$^2$ = {truncate(r_value, 2)}')
    plt.xlabel('Peak number')
    plt.ylabel('Mu from fit (ADCs)')
    plt.legend(fontsize=14, loc='upper left')
    plt.show()
    return slope, intercept

def fit_spectrum_and_plot(data, channel=7, initial_guess=[1000, 100, 20], bins=150, rng=(100,1500), num_peaks_fit=4, title=None):
    
    plt.figure(figsize=(7, 5))
    y, x, _    = plt.hist(data, bins=bins, range=rng, log=False, alpha=0.6)
    popt, pcov = scipy.optimize.curve_fit(multi_gaussian, shift_to_bin_centers(x), y, p0=initial_guess)
    
    plt.plot(x, multi_gaussian(x, *popt), 'r--', label='Fit')
    plt.xlabel('Amplitude (ADCs)', fontsize=15)
    plt.ylabel('Entries/bin',      fontsize=15)
    if title:
        plt.title(title, fontsize=15)
    else:
        plt.title(f"Spectrum for channel {channel} (height of the peaks)", fontsize=15)
    plt.tight_layout()
    plt.show()
    
    perr = np.sqrt(np.diag(pcov))
    
    all_means     = np.array([popt[i*3+1] for i in range(len(popt)//3)])
    all_means_err = np.array([perr[i*3+1] for i in range(len(perr)//3)])
    
    return plot_linear_fit(all_means[:num_peaks_fit], all_means_err[:num_peaks_fit])