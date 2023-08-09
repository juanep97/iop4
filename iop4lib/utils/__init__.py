from .filedproperty import *
from .plotting import *
from .sourcedetection import *
from .sourcepairing import *
from .astrometry import *

import os
import psutil
import numpy as np

import logging
logger = logging.getLogger(__name__)



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
        import numpy as np
        import scipy as sp
        import scipy.stats as stats

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
        res['mode'] = stats.mode(data, axis=None, keepdims=False).mode

        return res



def get_mem_children():
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
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss



def get_mem_parent_from_child():
    process = psutil.Process(os.getpid()).parent()
    mem_info = process.memory_info()
    return mem_info.rss



def get_mem_children_from_child():
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
    return get_mem_parent_from_child() + get_mem_children_from_child()











def get_target_fwhm_aperpix(redfL):
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
        xycen = centroid_quadratic(redf.mdata, *target.coord.to_pixel(redf.wcs), (15,15), search_boxsize=(5,5))
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

    return np.mean(fwhm_L), 3.0*np.mean(fwhm_L), 6.0*np.mean(fwhm_L), 9.0*np.mean(fwhm_L)
