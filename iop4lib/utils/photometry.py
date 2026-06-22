import re
import math
from io import StringIO
from importlib import resources

import numpy as np
import scipy as sp
import pandas as pd

from scipy.interpolate import (
    interp1d,
    RegularGridInterpolator,
)

import logging
logger = logging.getLogger(__name__)

from iop4lib.typing import *
    
def get_host_correction(astrosource, aperas, fwhm=None) -> tuple[float, float]:
    r""" Returns the contaminating flux and its uncertainty for a given astrosource and aperture radius. 
    
    If no correction is available for the given astrosource, it returns None, None.

    The value is interpolated from the tables in `host_correction_data.csv`. You can find
    more details in this file.

    Parameters
    ----------
    astrosource: AstroSource
        The astrosource for which to get the correction.
    aperas: float
        The aperture radius in arcsecs.
    fwhm: float, optional
        The fwhm in arcsec. Default is 7 arcsec (approximately this 
        fwhm for our 0.9m to 2.2m telescopes, doesnt affect much)

    Returns
    -------
    flux, flux_err: tuple[float, float]
        The contaminating flux and its uncertainty in mJy.

    or
    
    None, None

    """

    with open(resources.files("iop4lib.utils") / "host_correction_data.csv", 'r') as f:
        text = f.read()

    df = None

    def get_invariable_str(s):
        return s.replace(' ', '').replace('-','').replace('+','').replace('_','').upper()
    
    for table in text.split("#"*80):
        objname = re.findall(r"OBJECT: (.*)", table)[0]
        if get_invariable_str(astrosource.name) == get_invariable_str(objname) or \
            (astrosource.other_names is not None and any([get_invariable_str(other_name) == get_invariable_str(objname) for other_name in astrosource.other_names_list])):
            df = pd.read_csv(StringIO(table), comment="#", index_col=0)
            break

    if df is None:
        return None, None
    
    fluxes = df.copy()
    fluxes = fluxes.map(lambda x: x.split(" ")[0] if isinstance(x, str) and " " in x else x)

    uncerts = df.copy()
    uncerts = uncerts.map(lambda x: x.split(" ")[1] if isinstance(x, str) and " " in x else 0)

    aperture_grid = np.array(fluxes.index.values, dtype=float)

    if df.shape[1] > 2:

        if fwhm is None:
            raise Exception("Need the fwhm to apply host correction for this source")

        fwhm_grid = np.array(fluxes.columns.values, dtype=float)

        flux_interp = RegularGridInterpolator((aperture_grid, fwhm_grid), fluxes.values, fill_value=None, bounds_error=False)
        uncert_interp = RegularGridInterpolator((aperture_grid, fwhm_grid), uncerts.values, fill_value=None, bounds_error=False)

        flux = flux_interp((aperas, fwhm))
        flux_err = uncert_interp((aperas, fwhm))

    else:

        aperture_grid = np.array(df.index.values, dtype=float)

        flux_interp = interp1d(aperture_grid, fluxes.values.ravel(), bounds_error=False, fill_value="extrapolate")
        uncert_interp = interp1d(aperture_grid, uncerts.values.ravel(), bounds_error=False, fill_value="extrapolate")
        
        flux = flux_interp(aperas)
        flux_err = uncert_interp(aperas)

    logger.debug(f"Host correction for {astrosource.name} ({aperas=}, {fwhm=}) = {flux=}, {flux_err=}")

    return flux, flux_err

def zp_weighted_mean(zps, zps_err, rescale: Literal["no", "mle", "full", "1-sigma"] = "full"):
    """Estimate zero-point using a weighted mean.
    
    This function estimates the true zero-point using a weighted mean.
    Statistically, this is equivalent to assuming a Gaussian distribution of
    zero-point measurements, xi ~ N(mu, sigma_i^2)

    This function is sensitive to outliers. It also does not take into account 
    the possible scatter of zero-points due to (not-completely removed) 
    flat-fielding defects, or deviations in the reference magnitude of the 
    calibrators. For N>3; an outlier-robust method should be preferred.
    
    Since photometric uncertainties are often underestimated, this function can
    re-scale the uncertainty so that the model is forced to agree with the data.

    The rescaling options assume that the quoted uncertainties may be
    underestimated by a common multiplicative factor k >= 1,

        sigma_i,true = k * sigma_i.

    Because k is common to all points, the best-fit weighted mean mu is
    unchanged. Only the uncertainty on mu is inflated.

    If rescale is "no", no rescaling is performed:

        k = 1.

    If rescale is "mle", k is chosen as the constrained maximum-likelihood
    estimate for the Gaussian model with fitted mu and k >= 1:

        k^2 = max(1, chi2 / N).

    If rescale is "full", k is chosen from the usual reduced-chi2 estimate
    after fitting one parameter, mu:

        k^2 = max(1, chi2 / dof),

    where dof = N - 1. This makes the reduced chi2 no larger than 1.

    If rescale is "1-sigma", k is chosen so that the observed chi2 is no larger
    than the one-sided 1-sigma upper fluctuation expected from the chi2
    distribution:

        threshold = chi2_dof.ppf(Phi(1))
        k^2 = max(1, chi2 / threshold).

    The uncertainty is never down-scaled.
    """

    w = 1 / zps_err**2
    zp = np.sum(w*zps)/np.sum(w)
    zp_err = np.sqrt(1 / np.sum(w))

    # uncertainties can be understimated;
    # rescale so they agree with the model

    if rescale == "no":
        return zp, zp_err

    chi2 = np.sum(((zps - zp) / zps_err)**2)
    n = len(zps) 
    dof = n - 1
    rchi2 = chi2 / dof
    
    if rescale == "mle":
        k = np.sqrt(max(1,chi2/n))
    elif rescale == "full":
        k = np.sqrt(max(1,rchi2))
    elif rescale == "1-sigma":
        threshold = sp.stats.chi2.ppf(sp.stats.norm.cdf(1), dof)
        k = np.sqrt(max(1, chi2 / threshold))
    else:
        raise ValueError

    zp_err = zp_err * k

    return zp, zp_err, k

def zp_student(x, s, nu=3, rescale=True):
    r"""Robust estimation of zero-point using a Student-t distribution.

    Model
    -----
    For measurements x_i with uncertainties s_i, in the Gaussian sense, i.e.,
    such that

        P(|X_i - mu| < s_i) ~= 68% = 2*Phi(1)-1

    we assume x_i are sampled from a Student-t distribution t_nu(mu, a_i),
    where a_i = a_i(s_i) so that +/- s_i is still interpreted as the ~68%
    probability interval, i.e., we model X_i as

        X_i ~ t_nu(mu, a_i) = mu + a_i T_nu,

    and where a_i is the t-student scale, defined by those two relations, so

        P(|X_i - mu| < s_i) = P(|T_nu| < s_i/a_i) = 2*Phi(1)-1

    and therefore

        a_i = s_i / q_nu

    with

        q_nu = F_nu^{-1}(Phi(1))

    where F_nu is the CDF of the standard Student-t distribution with nu degrees
    of freedom.

    Equivalently:

        (X_i - mu) / a_i ~ T_nu

    The output zero-point is the maximum-likelihood estimate,

        zp = mu_est = argmin_mu nll(mu)

    where

        nll(mu) = -sum_i log p(x_i | mu, s_i, nu)

    The uncertainty is estimated from the likelihood profile,

        nll(mu_lo) = nll(mu_hi) = nll_min + 0.5

    and symmetrized as:

        mu_err = 0.5 * (mu_hi - mu_lo)

    This would be the exact 1-sigma interval for a Gaussian, and it is an 
    approximation in our case.

    Note:

    If rescale=True, uncertainties are assumed to be relative and possibly 
    underestimated, i.e. 
        s_i,true = k s_i
    with k>=1, and the fitted model is

        X_i = mu + k a_i T_nu,

    The log-likelihood for one point is

        log p(x_i | mu, k, s_i, nu)
            = log f_nu(r_i) + log(q_nu) - log(s_i) - log(k),

    with

        r_i = q_nu * (x_i - mu) / (k s_i).

    and k is fitted jointly with mu. The uncertainty on mu is then
    computed from the profile likelihood,

        nll_profile(mu) = min_{k >= 1} nll(mu, k).

    """
    
    x = np.asarray(x, float)
    s = np.asarray(s, float)

    qnu = sp.stats.t.ppf(sp.stats.norm.cdf(1), df=nu) # or qnu = 1 if you want just to treat s as the scales

    def nll(params):

        if rescale:
            mu, k = params
        else:
            mu = params[0]
            k = 1.0

        r = qnu * (x - mu) / (k * s)

        logp = (
            sp.stats.t.logpdf(r, df=nu)
            + np.log(qnu)
            - np.log(s)
            - np.log(k)
        )

        return -np.sum(logp)

    if rescale:
        kmin, kmax = 1.0, 1e3
        p0 = [np.median(x), 1.0]
        bounds = [(np.min(x), np.max(x)), (kmin, kmax)]
    else:
        p0 = [np.median(x)]
        bounds = [(np.min(x), np.max(x))]

    res = sp.optimize.minimize(
        nll,
        p0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not res.success:
        raise RuntimeError(res.message)

    mu = res.x[0]

    if rescale:
        k = res.x[1]
    else:
        k = 1.0
    
    nll_min = res.fun

    def nll_profile(mu_fixed):

        if not rescale:
            return nll([mu_fixed])

        kres = sp.optimize.minimize(
            lambda kk: nll([mu_fixed, kk[0]]),
            [k],
            method="L-BFGS-B",
            bounds=[(kmin, kmax)],
        )

        if not kres.success:
            raise RuntimeError(kres.message)

        return kres.fun

    target = nll_min + 0.5

    def delta(mu_fixed):
        return nll_profile(mu_fixed) - target

    def root(sign):
        edge = mu + sign * k * np.median(s)
        while delta(edge) < 0:
            edge = mu + 2 * (edge - mu)
        return sp.optimize.brentq(delta, min(mu, edge),max(mu, edge))

    mu_lo = root(-1)
    mu_hi = root(+1)

    mu_err = 0.5 * (mu_hi - mu_lo)

    if rescale:
        return mu, mu_err, k

    return mu, mu_err

class NoCalibratorsFound(Exception):
    pass

def calibrate_photopolresult(r: 'PhotoPolResult', photopolresults: List['PhotoPolResult']):
    """Calibrate PhotoPolResult `r` using the zero-points in photopolresults.
    
    If N<4 usable zero-points, the estimation will use a weighted mean.

    For N=4, it might use a weighted mean (with full re-escaling) if it has low 
    scatter, otherwise it will try to use an outlier-robust method.

    For N>4, it will try to use an outlier-robust method.
    
    If the robust-outlier method fails, it will fallback to the weighted mean.
    """
    
    calib_results = [
        c for c in photopolresults
        if (
            getattr(c.astrosource, f"mag_{r.band}", None)
            and r.astrosource in c.astrosource.calibrates.all()
        )
    ]

    zps = np.array([c.mag_zp for c in calib_results], dtype=float)
    zps_err = np.array([c.mag_zp_err for c in calib_results], dtype=float)

    idx = ~np.isnan(zps) & ~np.isnan(zps_err)

    zps = zps[idx]
    zps_err = zps_err[idx]

    if len(zps) == 0:
        raise NoCalibratorsFound(f"can not perform relative photometry on source {r.astrosource.name}, no calibrator zero-points found.")

    logger.debug(f"calibrating {r.astrosource.name} with {len(zps)} calibrators")
    
    try:
        zp1, dzp1, *_ = zp_weighted_mean(zps, zps_err)
    except Exception as e:
        zp1, dzp1 = None, None

    try:
        zp2, dzp2, *_ = zp_student(zps, zps_err)
    except Exception as e:
        zp2, dzp2 = None, None

    if len(zps) < 4 or (len(zps) == 4 and (dzp1 is not None and dzp1 < 0.05)) or dzp2 is None:
        zp, zp_err = zp1, dzp1
    else:
        zp, zp_err = zp2, dzp2
    
    # save the zp (to be) used
    r.mag_zp = zp
    r.mag_zp_err = zp_err

    # compute the calibrated magnitude
    r.mag = zp + r.mag_inst
    r.mag_err = math.sqrt(r.mag_inst_err**2 + zp_err**2)

    logger.debug(f"{r.mag=}, {r.mag_err=}")
