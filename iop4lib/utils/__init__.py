from .filedproperty import *
from .plotting import *
from .sourcedetection import *
from .sourcepairing import *
from .astrometry import *

import os
import psutil
import numpy as np
import scipy as sp
import scipy.stats 
import math
import warnings

import logging
logger = logging.getLogger(__name__)

import typing
from numpy.typing import NDArray
from typing import Sequence, Union, Tuple, Any
if typing.TYPE_CHECKING:
    from iop4lib.db import ReducedFit, AstroSource

def get_column_values(qs, column_names):
    """ Given a queryset and a list of column names, return a dictionary with the values of each column as a numpy array

    None values are converted to np.nan (to avoid numpy arrays having dtype=object) 
    """
    values_lists = zip(*qs.values_list(*column_names))
    values_lists = [[x if x is not None else np.nan for x in values] for values in values_lists]
    return {k: v for k, v in zip(column_names, map(np.array, values_lists))}


def qs_to_table(qs=None, model=None, data=None, column_names=None, default_column_names=None):
    """ Specify either queryset, or data and models."""

    from django.db import models

    if data is None or model is None:
        if qs is None:
            raise Exception("Either qs or data and model must be specified")
        else:
            model = qs.model
            data = qs.values()

    if column_names is None:
        column_names = [f.name for f in model._meta.get_fields() if hasattr(f, 'name') and f.name in data[0].keys()]

    if default_column_names is None:
        default_column_names = column_names

    columns = [{
                    "name": k, 
                    "title": model._meta.get_field(k).verbose_name, 
                    "visible": (k in default_column_names),
                    "type": "int" if isinstance(model._meta.get_field(k), models.IntegerField) else \
                            "float" if isinstance(model._meta.get_field(k), models.FloatField) else \
                            "str" if isinstance(model._meta.get_field(k), models.CharField) else \
                            "str" if isinstance(model._meta.get_field(k), models.TextField) else \
                            "date" if isinstance(model._meta.get_field(k), models.DateField) else \
                            "unknown",
                    "help": model._meta.get_field(k).help_text,
                } for k in column_names]
    
    return {'data': list(data), 'columns': columns}

def divisorGenerator(n):
    """Generator for divisors of n"""
    large_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


def stats_dict(data):
    """Return a dictionary with some basic statistics of the input data"""

    if isinstance(data, np.ma.MaskedArray):
        data = data.compressed()
    else:
            data = data.flatten()

    res = dict()

    res['mean'] = np.mean(data)
    res['median'] = np.median(data)
    res['std'] = np.std(data)
    res['min'] = np.min(data)
    res['max'] = np.max(data)
    res['mode'] = sp.stats.mode(data, axis=None, keepdims=False).mode

    return res


# Functions to get memory usage

def get_mem_children():
    """Return the sum of the memory usage of all children processes, from the parent process (in bytes)"""
    children = psutil.Process(os.getpid()).children(recursive=True)
    memory_usage = 0
    for child in children:
        try:
            mem_info = child.memory_info()
            memory_usage += mem_info.rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return memory_usage

def get_mem_current():
    """Return the memory usage of the current process (in bytes)"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def get_mem_parent_from_child():
    """Return the memory usage of the parent process, from one of the child processes (in bytes)"""
    process = psutil.Process(os.getpid()).parent()
    mem_info = process.memory_info()
    return mem_info.rss

def get_mem_children_from_child():
    """Return the sum of the memory usage of all children processes, from one of the child processes (in bytes)"""
    parent = psutil.Process(os.getpid()).parent()
    children = parent.children(recursive=True)
    memory_usage = 0
    for child in children:
        try:
            mem_info = child.memory_info()
            memory_usage += mem_info.rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return memory_usage

def get_total_mem_from_child():
    """Return the sum of the memory usage of all processes (parent and children) from one of the child processes (in bytes)"""
    return get_mem_parent_from_child() + get_mem_children_from_child()










def fit_fwhm(pos_px: tuple[float,float], data: NDArray = None, redf: 'ReducedFit' = None, px_max: int = None) -> float:
    r""" Fits a 1D gaussian + constant to the radial profile of the data around the given position, and returns the FWHM of the gaussian."""

    import numpy as np
    from photutils.profiles import RadialProfile
    from photutils.centroids import centroid_quadratic
    from astropy.modeling import models, fitting
    from iop4lib.instruments import Instrument
    from astropy.modeling.models import Moffat1D, Const1D, Gaussian1D

    if data is None:
        data = redf.mdata

    if px_max is None:
        # 0.4 arcsecs is excellent seeing, so let's consider 30 times that as the maximum radius
        px_max = int((30*0.4) / Instrument.by_name(redf.instrument).arcsec_per_pix)
        # this gives ~ 90 px for DIPOL, 30px for AndorT90

    pxs = np.arange(px_max)
    rp = RadialProfile(data, pos_px, pxs)

    fit = fitting.LevMarLSQFitter()

    # moffat = Moffat1D(x_0=0, amplitude=max(rp.profile)) + Const1D(min(rp.profile))
    # moffat[0].x_0.fixed = True
    # moffat_fit = fit(moffat, rp.radius, rp.profile)

    gaussian = Gaussian1D(amplitude=max(rp.profile), mean=0, stddev=1) + Const1D(min(rp.profile))
    gaussian[0].mean.fixed = True
    gaussian_fit = fit(gaussian, rp.radius, rp.profile)

    return gaussian_fit[0].fwhm

def fit_sigma(pos_px: tuple[float, float], *args, **kwargs) -> float:
    r""" Fits a 1D gaussian + constant to the radial profile of the data around the given position, and returns the standard deviation of the gaussian."""
    fwhm = fit_fwhm(pos_px, *args, **kwargs)
    sigma = fwhm / (2*np.sqrt(2*math.log(2)))
    return sigma



def fit_gaussian(px_start, redf=None, data=None, sigma_start=7, r_max=None, r_search=None):
    r""" Fits a 2D gaussian + constant to the data around the given position, and returns the fitted model.
    
    Parameters
    -----------
        px_start: (float, float)
            Initial pixel position of the center of the gaussian.
        sigma_start: int (default 7)
            Initial guess for the sigma of the gaussian.
        r_max: int (default 90)
            Region radius around px_start in which to perform the fit.
        r_search: int or float (default None)
            If provided, it will search for the maximum in a circle of radius r_search around px_start, and use that as the starting point for the fit.
    """

    import numpy as np
    from photutils.profiles import RadialProfile
    from astropy.modeling import fitting
    from iop4lib.instruments import Instrument
    from astropy.modeling.models import Const2D, Gaussian2D
    from iop4lib.instruments import Instrument

    if redf is not None:
        if data is None:
            data = redf.mdata

        if r_max is None:
            # 0.4 arcsecs is excellent seeing
            r_max = int((30*0.4) / Instrument.by_name(redf.instrument).arcsec_per_pix) 
    else:
        if r_max is None:
            r_max = 90

    height, width = data.shape

    x_start, y_start = px_start

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    
    if r_search is not None:
        idx_region = np.sqrt((X-x_start)**2 + (Y-y_start)**2) < r_search
        idx_region_max = np.argmax(data[idx_region])
        x_start, y_start = X[idx_region][idx_region_max], Y[idx_region][idx_region_max]

    idx_fit_region = np.sqrt((X-x_start)**2 + (Y-y_start)**2) < r_max

    X = X[idx_fit_region].flatten()
    Y = Y[idx_fit_region].flatten()
    Z = np.ma.array(data[idx_fit_region]).compressed()

    fit = fitting.LevMarLSQFitter()
    gaussian = Gaussian2D(amplitude=data[int(y_start), int(x_start)], x_mean=x_start, y_mean=y_start, x_stddev=sigma_start, y_stddev=sigma_start) + Const2D(np.median(Z))
    gaussian[0].x_stddev.tied = lambda model: model[0].y_stddev
    gaussian_fit = fit(gaussian, X, Y, Z)

    return gaussian_fit


def get_angle_from_history(redf: 'ReducedFit' = None, 
                           target_src: 'AstroSource' = None, 
                           calibrated_fits: Sequence['ReducedFit'] = None, 
                           n_max_fits=20) -> Union[float, float]:
    """ Compute the average rotation angle from a list of already calibrated fits.

    To compute it, it checks the WCS in the header of each calibrated fit.

    If no list of calibrated fits is given, but a reduced fit and a target source 
    are given, it will try to get it existing and calibrated reduced fits in the DB 
    for the same instrument and target source, but considering only photometry images.
    It will use at most n_max_fits (default 20) reduced fits, but will try to get the 
    ones closer in time to the given reduced fit. If a list of reduced fits is given,
    it will use that list instead of querying the DB, and n_max_fits will be ignored.

    If no target source is given, it will use the header_hintobject of the reduced fit.
    """
    from iop4lib.enums import IMGTYPES, OBSMODES
    from iop4lib.db import ReducedFit
    from iop4lib.instruments import Instrument

    if calibrated_fits is None:
        if target_src is None:
            target_src = redf.header_hintobject

        qs = ReducedFit.objects.filter(instrument=redf.instrument, 
                                                imgtype=IMGTYPES.LIGHT, 
                                                obsmode=OBSMODES.PHOTOMETRY, 
                                                flags__has=ReducedFit.FLAGS.BUILT_REDUCED,
                                                sources_in_field__in=[target_src])
        
        if len(qs) == 0:
            logger.warning(f"No calibrated fits for {redf.instrument} {target_src.name}, using other sources too")
            qs = ReducedFit.objects.filter(instrument=redf.instrument, 
                                                imgtype=IMGTYPES.LIGHT, 
                                                obsmode=OBSMODES.PHOTOMETRY, 
                                                flags__has=ReducedFit.FLAGS.BUILT_REDUCED)
        if len(qs) < 5:
            logger.warning(f"Less than 5 calibrated fits for {redf.instrument} {target_src.name}, using all of them")

        jds = np.array(qs.values_list('juliandate', flat=True))
        idx = np.argsort(np.abs(jds - redf.juliandate))[0:n_max_fits]
        calibrated_fits = [qs[int(i)] for i in idx]

    angle_L = list()
    for calibrated_fit in calibrated_fits:
        try:
            w = WCS(calibrated_fit.header, key="A")
        except Exception as e: # usually KeyError
            logger.warning(f"Could not get WCS 'A' from ReducedFit {calibrated_fit.id}: {e}")
            continue
            
        
        # Extract the PC matrix elements
        pc_11, pc_12 = w.wcs.pc[0]
        pc_21, pc_22 = w.wcs.pc[1]
        
        # Calculate the rotation angle in degrees
        angle = np.degrees(np.arctan2(pc_21, pc_11)) % 360 # wrap angles at 0,360 (-180 = 180)

        if 'FLIPSTAT' in redf.rawfit.header and redf.rawfit.header['FLIPSTAT'] == 'Flip':
                angle = - angle

        angle_L.append(angle)

    if len(angle_L) == 0:
        logger.warning(f"No angle found in history for {redf.instrument} {target_src.name}")

        if hasattr(Instrument.by_name(redf.instrument), "default_sky_angle"):
            logger.warning(f"Using default sky angle for {redf.instrument}")
            return Instrument.by_name(redf.instrument).default_sky_angle, Instrument.by_name(redf.instrument).default_sky_angle_std
        else:
            logger.error(f"No default sky angle for {redf.instrument}, returning NaN")
            return np.nan, np.nan
    
    angle_mean = np.mean(angle_L)
    angle_std = np.std(angle_L)

    return angle_mean, angle_std

def build_wcs_centered_on(target_px: tuple[float, float],
                          target_coord: tuple[float, float] = None,
                          target_src : 'AstroSource' = None, 
                          redf: 'ReducedFit' = None, 
                          angle: float = None, 
                          pixel_scale: float = None) -> WCS:
    r""" Build a WCS object with the target source at the target pixel position.

    Builds a WCS object that has the target astrosource at the given pixel position in the image,
    with the given angle (in degrees) and pixel scale (in degrees/pixel).

    If either angle, pixel_scale or target source is not given, it will try to get them
    from the given reduced fit.

    If no pixel scale is given, it will use the pixel scale of the instrument of the 
    reduced fit.

    If no angle is given, it will try to get it from the history of the reduced fits for the same 
    instrument and target source, but considering only photometry images.

    If no target source is given, it will use the header_hintobject of the reduced fit.
    """

    from iop4lib.instruments import Instrument

    if angle is None:
        angle, angle_std = get_angle_from_history(redf, target_src) # in degrees
        if angle_std > 0.5:
            logger.warning(f"Large angle std: {angle=}, {angle_std=}")
    
    if pixel_scale is None:
        pixel_scale = Instrument.by_name(redf.instrument).arcsec_per_pix / 3600 # in degrees

    if target_src is None and redf is not None:
        target_src = redf.header_hintobject

    if target_coord is None and target_src is not None:
        known_ra, known_dec = target_src.coord.ra.deg, target_src.coord.dec.deg, 
    else:
        known_ra, known_dec = target_coord.ra.deg, target_coord.dec.deg

    known_x_pixel, known_y_pixel = target_px[0], target_px[1]

    w = WCS(naxis=2)

    # Set RA, DEC for reference pixel (in degrees)
    w.wcs.crval = [known_ra, known_dec]

    # Set reference pixel
    w.wcs.crpix = [known_x_pixel, known_y_pixel] 

    # Other settings remain the same
    w.wcs.cdelt = np.array([pixel_scale, pixel_scale])
    w.wcs.pc = [[+np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [+np.sin(np.radians(angle)), +np.cos(np.radians(angle))]]

    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return w



@dataclasses.dataclass
class SimbadSource():
    name: str
    ra_hms: str
    dec_dms: str
    otype: str
    other_names: str = None


    def __str__(self):
        return f"<Simbad source {self.name}>"

    @property
    def coord(self):
        return SkyCoord(self.ra_hms, self.dec_dms, unit=(u.hourangle, u.deg))
    
    def is_in_field(self, wcs, height, width):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, y = self.coord.to_pixel(wcs)
        except:
            return False
        else:
            if (0 <= x < height) and (0 <= y < width):
                return True
            else:
                return False
            

def get_simbad_table(center_coord, radius, Nmax=6, all_types=False):
    from astroquery.simbad import Simbad
    Simbad.ROW_LIMIT = 100
    Simbad.add_votable_fields('otype')
    Simbad.add_votable_fields('flux(R)', 'flux(G)')

    tb_simbad = Simbad.query_region(center_coord, radius=radius, epoch="J2000")

    if not all_types:
        otypes = ["Star", "QSO", "BLLac", "BLLac"]

        idx = np.full(len(tb_simbad), fill_value=False)
        for otype in otypes:
            idx = idx | (tb_simbad["OTYPE"] == otype)

        tb_simbad = tb_simbad[idx]

    if len(tb_simbad) > Nmax:
        tb_simbad = tb_simbad[0:Nmax]

    return tb_simbad

def get_simbad_sources(center_coord, radius, Nmax=6, all_types=False, exclude_self=True):

    tb_simbad = get_simbad_table(center_coord, radius, Nmax=6, all_types=all_types)

    simbad_sources = list()

    for row in tb_simbad:
        name = row['MAIN_ID']
        coord = SkyCoord(Angle(row['RA'], unit=u.hourangle), Angle(row['DEC'], unit=u.degree), frame='icrs')
        ra_hms = coord.ra.to_string(unit=u.hourangle, sep=':', precision=4, pad=True)
        dec_dms = coord.dec.to_string(unit=u.degree, sep=':', precision=4, pad=True)
        otype = row['OTYPE']

        simbad_sources.append(SimbadSource(name=name, ra_hms=ra_hms, dec_dms=dec_dms, otype=otype))

    if exclude_self:
        simbad_sources = [simbad_source for simbad_source in simbad_sources if simbad_source.coord.separation(center_coord).arcsec > 1]

    return simbad_sources




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

    import numpy as np
    import pandas as pd
    from io import StringIO
    from importlib import resources
    import re

    with open(resources.files("iop4lib.utils") / "host_correction_data.csv", 'r') as f:
        text = f.read()

    df = None

    def get_invariable_str(s):
        return s.replace(' ', '').replace('-','').replace('+','').replace('_','').upper()
    
    for table in text.split("#"*80):
        objname = re.findall(r"OBJECT: (.*)", table)[0]
        if get_invariable_str(astrosource.name) == get_invariable_str(objname) or \
            (astrosource.other_names is not None and any([get_invariable_str(other_name) == get_invariable_str(objname) for other_name in astrosource.other_names_list])):
            df = pd.read_csv(StringIO(table), comment="#")
            break

    if df is None:
        return None, None
    
    if fwhm is None:
        fwhm = 7 

    # Get the column whose header is closest to the indicated fwhm
    
    fwhm_L = np.array([float(fwhm) for fwhm in df.columns[1:]])
    fwhm_idx = np.argmin(np.abs(fwhm_L - fwhm))

    # Get the values and uncertainties of the contaminating flux for the indicated fwhm

    flux_L, flux_err_L = zip(*[v.split(" ") for v in df.iloc[:,fwhm_idx+1]])

    flux_L = np.array([float(flux) for flux in flux_L])
    flux_err_L = np.array([float(flux_err) for flux_err in flux_err_L])

    aperas_L = np.array([float(apera) for apera in df.iloc[:,0]])

    # Interpolate the contaminating flux and its uncertainty for the indicated aperture radius
    # Use simple linear interpolation from np.interp:

    flux = np.interp(aperas, aperas_L, flux_L)
    flux_err = np.interp(aperas, aperas_L, flux_err_L)

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


def calibrate_photopolresult(result, photopolresult_L):

    # 3.a Average the zero points

    # get all the computed photopolresults that calibrate this source
    calibrator_results_L = [r for r in photopolresult_L if r.astrosource.calibrates.filter(pk=result.astrosource.pk).exists()]

    # Create an array with nan instead of None (this avoids the dtype becoming object)
    calib_mag_zp_array = np.array([r.mag_zp or np.nan for r in calibrator_results_L if r.astrosource.is_calibrator]) 
    calib_mag_zp_array_err = np.array([r.mag_zp_err or np.nan for r in calibrator_results_L if r.astrosource.is_calibrator])

    idx = (~np.isnan(calib_mag_zp_array)) & (~np.isnan(calib_mag_zp_array_err))
    calib_mag_zp_array = calib_mag_zp_array[idx]
    calib_mag_zp_array_err = calib_mag_zp_array_err[idx]

    if len(calib_mag_zp_array) == 0:
        logger.error(f"can not perform relative photometry on source {result.astrosource.name}, no calibrator zero-points found.")
        return

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