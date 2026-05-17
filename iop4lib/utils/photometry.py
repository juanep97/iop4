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

import typing
    
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



def filter_zero_points(calib_mag_zp_array, calib_mag_zp_array_err):

    zp_avg = np.nanmean(calib_mag_zp_array)
    zp_std = np.nanstd(calib_mag_zp_array)

    logger.debug(f"{zp_avg=}, {zp_std=}")

    # if there are enough calibrators, try remove those with a higher error
    # sometimes one of them is bad in a image, and if included, the error is too high
    if len(calib_mag_zp_array) > 5:
        zp_err_avg = np.nanmean(calib_mag_zp_array_err)
        zp_err_std = np.nanstd(calib_mag_zp_array_err)

        logger.debug(f"{zp_err_avg=}, {zp_err_std=}")

        idx = calib_mag_zp_array_err < zp_err_avg + 1*zp_err_std
        filtered_calib_mag_zp_array = calib_mag_zp_array[idx]
        filtered_calib_mag_zp_array_err = calib_mag_zp_array_err[idx]

        n_filtered_on_error = len(calib_mag_zp_array) - len(filtered_calib_mag_zp_array)

        if len(filtered_calib_mag_zp_array) >= 3:
            logger.debug(f"sigma clipped {n_filtered_on_error} calibrators based on their error")

            calib_mag_zp_array = filtered_calib_mag_zp_array
            calib_mag_zp_array_err = filtered_calib_mag_zp_array_err

            logger.debug(f"{calib_mag_zp_array=}")
            logger.debug(f"{calib_mag_zp_array_err=}")

            zp_avg = np.nanmean(calib_mag_zp_array)
            zp_std = np.nanstd(calib_mag_zp_array)

            logger.debug(f"{zp_avg=}, {zp_std=}")
            logger.debug(f"{zp_err_avg=}, {zp_err_std=}")

    # next, try to remove those that are too far from the average
    if len(calib_mag_zp_array) > 5:
        idx = abs(calib_mag_zp_array - zp_avg) < 1*zp_std
        calib_mag_zp_array_sigma_clipped = calib_mag_zp_array[idx]
        calib_mag_zp_array_err_sigma_clipped = calib_mag_zp_array_err[idx]

        n_filtered_on_value = len(calib_mag_zp_array) - len(calib_mag_zp_array_sigma_clipped)

        if len(calib_mag_zp_array_sigma_clipped) >= 3:
            logger.debug(f"sigma cipped {n_filtered_on_value} calibrators based on their value")

            calib_mag_zp_array = calib_mag_zp_array_sigma_clipped
            calib_mag_zp_array_err = calib_mag_zp_array_err_sigma_clipped

            logger.debug(f"{calib_mag_zp_array=}")
            logger.debug(f"{calib_mag_zp_array_err=}")

    zp_avg = np.nanmean(calib_mag_zp_array)
    zp_std = np.nanstd(calib_mag_zp_array)

    logger.debug(f"{zp_avg=}, {zp_std=}")

    zp_err = np.sqrt(np.nansum(calib_mag_zp_array_err ** 2)) / len(calib_mag_zp_array_err)
    zp_err = math.sqrt(zp_err ** 2 + zp_std ** 2)

    return calib_mag_zp_array, calib_mag_zp_array_err, zp_avg, zp_std, zp_err



class NoCalibratorsFound(Exception):
    pass

def calibrate_photopolresult(result, photopolresult_L):

    # 3.a Average the zero points

    # get all the computed photopolresults that calibrate this source
    # calibrator_results_L = [r for r in photopolresult_L if r.astrosource.calibrates.filter(pk=result.astrosource.pk).exists()]
    calibrator_results_L = [r for r in photopolresult_L if getattr(r.astrosource, f"mag_{r.band}", None)]

    # Create an array with nan instead of None (this avoids the dtype becoming object)
    calib_mag_zp_array = np.array([r.mag_zp or np.nan for r in calibrator_results_L if r.astrosource.is_calibrator]) 
    calib_mag_zp_array_err = np.array([r.mag_zp_err or np.nan for r in calibrator_results_L if r.astrosource.is_calibrator])

    idx = (~np.isnan(calib_mag_zp_array)) & (~np.isnan(calib_mag_zp_array_err))
    calib_mag_zp_array = calib_mag_zp_array[idx]
    calib_mag_zp_array_err = calib_mag_zp_array_err[idx]

    if len(calib_mag_zp_array) == 0:
        raise NoCalibratorsFound(f"can not perform relative photometry on source {result.astrosource.name}, no calibrator zero-points found.")

    logger.debug(f"calibrating {result.astrosource.name} with {len(calib_mag_zp_array)} calibrators")
    logger.debug(f"{calib_mag_zp_array=}")
    logger.debug(f"{calib_mag_zp_array_err=}")

    calib_mag_zp_array, calib_mag_zp_array_err, zp_avg, zp_std, zp_err = filter_zero_points(calib_mag_zp_array, calib_mag_zp_array_err)

    # 3.b Compute the calibrated magnitude

    # save the zp (to be) used
    result.mag_zp = zp_avg
    result.mag_zp_err = zp_err

    # compute the calibrated magnitude
    result.mag = zp_avg + result.mag_inst
    result.mag_err = math.sqrt(result.mag_inst_err**2 + zp_err**2)

    logger.debug(f"{result.mag=}, {result.mag_err=}")
