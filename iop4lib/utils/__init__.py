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

import logging
logger = logging.getLogger(__name__)


def get_column_values(qs, column_names):
    """ Given a queryset and a list of column names, return a dictionary with the values of each column as a numpy array

    None values are converted to np.nan (to avoid numpy arrays having dtype=object) 
    """
    values_lists = zip(*qs.values_list(*column_names))
    values_lists = [[x if x is not None else np.nan for x in values] for values in values_lists]
    return {k: v for k, v in zip(column_names, map(np.array, values_lists))}



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
    """Return the sum of the memory usage of all children processes, from the parent process"""
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
    """Return the memory usage of the current process"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def get_mem_parent_from_child():
    """Return the memory usage of the parent process, from one of the child processes"""
    process = psutil.Process(os.getpid()).parent()
    mem_info = process.memory_info()
    return mem_info.rss

def get_mem_children_from_child():
    """Return the sum of the memory usage of all children processes, from one of the child processes"""
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
    """Return the sum of the memory usage of all processes (parent and children) from one of the child processes"""
    return get_mem_parent_from_child() + get_mem_children_from_child()









# Function to get target FWHM

def get_target_fwhm_aperpix(redfL, reductionmethod=None):
    r"""estimate an appropriate common aperture for a list of reduced fits.
    
    It fits the target source profile in the fields and returns some multiples of the fwhm which are used as the aperture and as the inner and outer radius of the annulus for local bkg estimation).
    """

    import numpy as np
    from iop4lib.db import AstroSource
    from photutils.profiles import RadialProfile
    from photutils.centroids import centroid_quadratic
    from astropy.modeling.models import Moffat1D, Const1D, Gaussian1D
    from astropy.modeling import models, fitting

    astrosource_S = set.union(*[set(redf.sources_in_field.all()) for redf in redfL])
    target_L = [astrosource for astrosource in astrosource_S if astrosource.srctype != AstroSource.SRCTYPES.CALIBRATOR]

    logger.debug(f"{astrosource_S=}")

    if len(target_L) > 0:
        target = target_L[0]
    elif len(astrosource_S) > 0:
        target = astrosource_S.pop()
    else:
        return np.nan, np.nan, np.nan, np.nan
  
    fwhm_L = list()

    for redf in redfL:
        try:
            xycen = centroid_quadratic(redf.mdata, *target.coord.to_pixel(redf.wcs), (15,15), search_boxsize=(5,5))
        except Exception as e:
            logger.warning(f"centroid_quadatric failed: {e}. Using target.coord.to_pixel(redf.wcs), probably wrong")
            xycen = target.coord.to_pixel(redf.wcs)

        if not all(np.isfinite(xycen)):
            xycen = target.coord.to_pixel(redf.wcs)

        pxs = np.arange(30)

        rp = RadialProfile(redf.mdata, xycen, pxs)

        fit = fitting.LevMarLSQFitter()

        # moffat = Moffat1D(x_0=0, amplitude=max(rp.profile)) + Const1D(min(rp.profile))
        # moffat[0].x_0.fixed = True
        # moffat_fit = fit(moffat, rp.radius, rp.profile)

        gaussian = Gaussian1D(amplitude=max(rp.profile), mean=0, stddev=1) + Const1D(min(rp.profile))
        gaussian[0].mean.fixed = True
        gaussian_fit = fit(gaussian, rp.radius, rp.profile)

        logger.debug(f"{target.name}: Gaussian FWHM: {gaussian_fit[0].fwhm:.1f} px")
        # logger.debug(f"{target.name}: Moffat FWHM: {moffat_fit[0].fwhm:.1f} px")

        # list.append(fwhm_L, (moffat_fit[0].fwhm+gaussian_fit[0].fwhm)/2)

        fwhm_L.append(gaussian_fit[0].fwhm)

    mean_fwhm = np.mean(fwhm_L)
    sigma = mean_fwhm / (2*np.sqrt(2*math.log(2)))
    r = sigma
    
    return mean_fwhm, 5.0*r, 15.0*r, 20.0*r
