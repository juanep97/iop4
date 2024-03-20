import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)  

# other imports 

import os
import gc
from pathlib import Path
import numpy as np
import scipy as sp
import matplotlib as mplt
import matplotlib.pyplot as plt
import astrometry
import astropy.io.fits as fits
from astropy.wcs import WCS
from iop4lib.telescopes import Telescope
import astropy.units as u
from astropy.coordinates import SkyCoord
import multiprocessing
import functools
import itertools
import dataclasses

# iop4lib imports

from iop4lib.utils.sourcepairing import (get_pairs_d, get_pairs_dxy, get_best_pairs)
from iop4lib.utils.sourcedetection import (get_bkg, get_segmentation, get_cat_sources_from_segment_map)
from iop4lib.utils.plotting import build_astrometry_summary_images

# logging

import logging
logger = logging.getLogger(__name__)

import typing
if typing.TYPE_CHECKING:
    from iop4lib.db.reducedfit import ReducedFit

@dataclasses.dataclass
class BuildWCSResult():
    r"""
        'success': bool
            whether the appropiate WCSs was built successfully
        'wcslist': list 
            list of WCS objects built (usually one, two if there are extraordinary sources in the image)
        'info': dict or None 
            dict with extra information about the process

        Boolean evaluation of this object returns the value of 'success'.
    """
    success: bool
    wcslist: list[WCS] = dataclasses.field(default_factory=list)
    info: dict = dataclasses.field(default_factory=dict)   

    def __bool__(self):
        return self.success 


def build_wcs_params_shotgun(redf: 'ReducedFit', shotgun_params_kwargs : dict = None, hard : bool = False, summary_kwargs : dict = {'build_summary_images':True, 'with_simbad':True}) -> BuildWCSResult:
    """ Build the appropiate WCSs for a ReducedFit image, trying different parameters. See `build_wcs` for more info.

    Note: at the moment, this function tries source extraction with different combination of parameters and thresholds for 
    source extraction by calling a helper func (`_build_wcs_detect_and_try_solve`) with these parameters, which detects 
    the  sources with `photutils` image segmentation and tries to solve the WCS with the `astrometry.net` python wrapper. 
    The parameter combinations are chosen depending on the exposure time and the presence of pairs in the image.
    
    TODO:

    - Implement a more robust way to choose the parameters for source extraction such that the astrometry solver works  with less 
      attempts.
    - Explore other detectors and solvers if necessary to improve speed, sucess rate and accuracy.
    - Use pre-computed pair distances.
    """

    param_dicts_L = []

    params = dict()

    # Define the possible values of each parameter to `_build_wcs_detect_and_try_solve`

    # SENSIBLE VERSION

    # params["keep_n_seg"] = [150] # it is useless to try with much more than ~100 sources, if it is solvable it will be with the brightest sources, solver will take much much time otherwise
    # params["border_margin_px"] = [20] # sources very close to the border could be due to border effects
    # params["bkg_filter_size"] = [11, 7, 5, 3] # the higher the better, it will make the bkg much smoother (but a kernel too high will be very slow or get in computational errors, 11 is more than enough, 7 also produces pretty smooth results. Sometimes, smaller values (e.g. 5 or even 3 are better to capture very local variations). Bkg will be interpolated anyway according to bkg_box_size.
    # params["bkg_box_size"] = [8, 16, 32, 64] # 16 is low but produces nice backgrounds, anyway this is for source extraction, not for photometry, where higher sizes e.g. 32, 64, or even more would be more sensible. This is later interpolated, so with 16 you manage to capture more local variations of the background, important when there are sources of very different brightness in the image.
    # params["seg_fwhm"] = [1.0, 8.0] # the lower the better. Higher fwhm (8.0) will discard many sources due to noise, but will decrease precision. This is specially bad for images with pairs, as a pair might be detected as one source only in the middle.
    # params["seg_kernel_size"] = [None] # automatically set according to fwhm, default is 2*int(fwhmw)+1, which is good enough.
    # params["npixels"] = [64, 32, 16, 8] # higher will discard more fake sources, but will also discard real sources. However the larger, the centroid position might be more noisy. Better to keep it 8-64 depending on need (pairs vs no pairs, crowded vs no crowded, low exp vs high exp).
    # params["allsky"] = [False] # whether to look over all sky, sometimes the hint keywords are wrong (but vary rarely).
    # params["output_logodds_threshold"] = [80, 40, 21] # 21 is the default of the astrometry.net solver. Very rarely with very low exp. / very little sources, we might ned to put it to 14 to get a good match. LOWER IS NOT RECOMMENDED, as it might generate wrong matches.
    # params["n_rms_seg"] = [24.0, 12.0, 6.0, 3.0, 1.5, 1.0, 0.66] # might be worth it to set it depending on exposure time, very rarely even lower values might be helpful (0.5, 0.4).

    # FAST VERSION

    if redf.has_pairs:
        params["keep_n_seg"] = [300]
    else:
        params["keep_n_seg"] = [150]

    params["border_margin_px"] = [5]

    if redf.exptime <= 0.05:
        params["output_logodds_threshold"] = [14]
        params["n_rms_seg"] = [1.0, 0.8, 0.66]
    elif 0.05 < redf.exptime <= 0.1:
        params["output_logodds_threshold"] = [14]
        params["n_rms_seg"] = [1.2, 1.0, 0.8]
    elif 0.1 < redf.exptime <= 10.0:
        params["output_logodds_threshold"] = [14]
        params["n_rms_seg"] = [1.5, 1.2, 1.0]
    elif 10.0 < redf.exptime <= 30.0:
        params["output_logodds_threshold"] = [14]
        params["n_rms_seg"] = [3.0, 1.5, 1.2]
    else:
        params["output_logodds_threshold"] = [14]
        params["n_rms_seg"] = [6.0, 3.0, 1.5]

    params["bkg_filter_size"] = [11, 3] 
    params["bkg_box_size"] = [16, 8]
    params["seg_fwhm"] = [1.0]
    params["seg_kernel_size"] = [None]
    params["npixels"] = [32, 8, 16]
    params["allsky"] = [False]

    ## Substitute default params combinations with specified ones

    if shotgun_params_kwargs is not None:
        for k, v in shotgun_params_kwargs.items():
            params[k] = v

    ## Build a list of dictionaries with all the possible combinations of parameters

    param_dicts_L_light = []
    for values in itertools.product(*params.values()):
        param_dict = dict(zip(params.keys(), values))
        param_dicts_L_light.append(param_dict)

    ## filter out some senseless combinations

    param_dicts_L_light = list(filter(lambda x: x['bkg_filter_size'] < x['bkg_box_size'], param_dicts_L_light)) 

    ## add combinations to the list

    param_dicts_L = param_dicts_L + param_dicts_L_light 
    
    # HARD VERSION

    if hard:
        params["keep_n_seg"] = [300, 150]
        params["border_margin_px"] = [5, 20]
        params["bkg_filter_size"] = [11, 7, 5, 3]
        params["bkg_box_size"] = [64, 32, 16, 8]
        params["seg_fwhm"] = [1.0, 2.0, 4.0, 8.0]
        params["seg_kernel_size"] = [None]
        params["npixels"] = [64, 32, 16, 8]
        params["allsky"] = [False]
        params["output_logodds_threshold"] = [21, 14]
        params["n_rms_seg"] = [6.0, 3.0, 1.5, 1.0, 0.8, 0.66]

        if shotgun_params_kwargs is not None:
            for k, v in shotgun_params_kwargs.items():
                params[k] = v

        param_dicts_L_hard = []
        for values in itertools.product(*params.values()):
            param_dict = dict(zip(params.keys(), values))
            if param_dict not in param_dicts_L:
                param_dicts_L_hard.append(param_dict)

        param_dicts_L_hard = list(filter(lambda x: x['bkg_filter_size'] < x['bkg_box_size'], param_dicts_L_hard)) ## filter out some senseless combinations

        np.random.shuffle(param_dicts_L_hard) #inplace, so we try very different combinations first
        
        param_dicts_L = param_dicts_L + param_dicts_L_hard

    # Attempt to build the wcs with each combination of parameters

    logger.debug(f"{redf}: {len(param_dicts_L)} different combinations of parameters to try.")

    for i, params_dict in enumerate(param_dicts_L):
        logger.debug(f"{redf}: attempt {i+1} / {len(param_dicts_L)}, ({params_dict}) ...")

        build_wcs_result = _build_wcs_params_shotgun_helper(redf, **params_dict)
        # try: # if we want to allow exceptions, but if there's an error something is wrong, we should fix the cause of the underlying exception; different thing is that we could not solve astrometry
        #     build_wcs_result = _build_wcs_params_shotgun_helper(redf, **params_dict)
        # except Exception as e:
        #     logger.error(f"{redf}: some error ocurred during attempt {i+1} / {len(param_dicts_L)}, ({params_dict}), ignoring. Error: {e}")
        #     build_wcs_result = {'success': False}

        if build_wcs_result.success:
            logger.debug(f"{redf}: WCS built with attempt {i+1} / {len(param_dicts_L)} ({params_dict}).")
            break
    else:
        # if none worked
        logger.error(f"{redf}: could not solve astrometry with any of the {len(param_dicts_L)} default parameter combinations for source extraction.")
        return BuildWCSResult(success=False)

    # add the parameters that worked to the result
    build_wcs_result.info['params'] = params_dict

    # build summary images
    if summary_kwargs['build_summary_images']:
        logger.debug(f"{redf}: building summary images.")
        build_astrometry_summary_images(redf, build_wcs_result.info, summary_kwargs=summary_kwargs)

    # to remove unwanted info from the result
    to_save_from_info_kw_L = ['params', 'bm', 'seg_d0', 'seg_disp_sign', 'seg_disp_xy', 'seg_disp_sign_xy', 'seg_disp_xy_best']
    build_wcs_result.info = {k:build_wcs_result.info[k] for k in to_save_from_info_kw_L if k in build_wcs_result.info}
    build_wcs_result.info['logodds'] = build_wcs_result.info['bm'].logodds
    build_wcs_result.info['method'] = 'shotgun'

    return build_wcs_result




def _build_wcs_params_shotgun_helper(redf, has_pairs=None,
        bkg_filter_size = 11,
        bkg_box_size = 16,
        seg_kernel_size = None,
        npixels = 32,
        seg_fwhm = 1.0,
        n_rms_seg = 1.0,
        keep_n_seg = 200,
        border_margin_px = 20,
        dx_eps=None,
        dy_eps=None,
        d_eps=None,
        dx_min=None,
        dx_max=None,
        dy_min=None,
        dy_max=None,
        d_min=None,
        d_max=None,
        bins=None,
        hist_range=None,
        position_hint=None, size_hint=None, allsky=False,
        output_logodds_threshold=21) -> BuildWCSResult:
    """ helper func, see build_wcs_params_shotgun for more info. """

    imgdata = redf.mdata

    if has_pairs is None:
        has_pairs = redf.has_pairs

    if size_hint is None:
        size_hint = redf.get_astrometry_size_hint()

    if position_hint is None:
        position_hint = redf.get_astrometry_position_hint(allsky=allsky)

    if has_pairs:
        if bins is None:
            bins = int( 0.75 * max(imgdata.shape) )
        if hist_range is None:
            hist_range = (0, min(imgdata.shape))

    # Background substraction

    bkg = get_bkg(imgdata, filter_size=bkg_filter_size, box_size=bkg_box_size)
    imgdata_bkg_substracted = imgdata - bkg.background

    # Image Segmentation

    seg_threshold = n_rms_seg * bkg.background_rms
    segment_map, convolved_data = get_segmentation(imgdata_bkg_substracted, fwhm=seg_fwhm, kernel_size=seg_kernel_size, npixels=npixels, threshold=seg_threshold)

    if segment_map is None:
        logger.debug(f"{redf}: No segments found, returning early.")
        return BuildWCSResult(success=False)
    
    seg_cat, pos_seg, tb = get_cat_sources_from_segment_map(segment_map, imgdata_bkg_substracted, convolved_data)
    
    logger.debug(f"{redf}: {len(pos_seg)=}")

    if keep_n_seg is not None and len(pos_seg) > keep_n_seg:
        logger.debug(f"{redf}: Keeping only {keep_n_seg} brightest segments.")
        pos_seg = pos_seg[:keep_n_seg]

    if border_margin_px is not None:
        logger.debug(f"{redf}: Removing segments within {border_margin_px} px from border.")
        pos_seg = [pos for pos in pos_seg if ( (border_margin_px < pos[0] < imgdata.shape[1]-border_margin_px) and (border_margin_px < pos[1] < imgdata.shape[0]-border_margin_px))]

    # Pair finding with results from image segmentation
    
    if has_pairs:
        seg1, seg2, seg_d0, seg_disp_sign = get_pairs_d(pos_seg, bins=bins, hist_range=hist_range, d_min=d_min, d_eps=d_eps, d_max=d_max)
        logger.debug(f"{redf}: seg pairs -> {len(seg1)} ({len(seg1)/len(pos_seg)*100:.1f}%), seg_disp_sign={seg_disp_sign}")
        seg1_best, seg2_best, seg_disp_best, seg_disp_sign_best = get_best_pairs(seg1, seg2, seg_disp_sign)
        logger.debug(f"{redf}: seg pairs best -> {len(seg1_best)} ({len(seg1_best)/len(pos_seg)*100:.1f}%), seg_disp_sign_best={seg_disp_sign_best}")
        seg1xy, seg2xy, seg_disp_xy, seg_disp_sign_xy = get_pairs_dxy(pos_seg, bins=bins, hist_range=hist_range, dx_min=dx_min, dx_max=dx_max, dy_min=dy_min, dy_max=dy_max, dx_eps=dx_eps, dy_eps=dy_eps)
        logger.debug(f"{redf}: seg pairs xy -> {len(seg1xy)}, disp_sign_xy={seg_disp_sign_xy}")
        seg1xy_best, seg2xy_best, seg_disp_xy_best, seg_disp_sign_xy_best = get_best_pairs(seg1xy, seg2xy, seg_disp_sign_xy)
        logger.debug(f"{redf}: seg pairs xy best -> {len(seg1xy_best)} ({len(seg1xy_best)/len(pos_seg)*100:.1f}%), seg_disp_sign_xy_best={seg_disp_sign_xy_best}")

    # Solve astrometry 

    bm = None

    if has_pairs: ## Attempt both with D pairs and XY pairs
        attempts = ((f"Seg Best XY Pairs (n={len(seg1xy_best)})", seg1xy_best, seg_disp_sign_xy_best),
                    (f"Seg Best D Pairs (n={len(seg1_best)})", seg1_best, seg_disp_sign_best),)
    else: ## Use the positions of the segments
        attempts = ((f"Seg Pos (n={len(pos_seg)})", pos_seg, None),)
    
    for msg, stars, disp_sign in attempts:
        if len(stars) == 0:
            logger.debug(f"{redf}: can not solve astrometry with {msg} because no pairs where found.")
            continue

        logger.debug(f"{redf}: trying to solve astrometry with {msg} ({output_logodds_threshold=}).")

        solution = solve_astrometry(stars, size_hint=size_hint, position_hint=position_hint, output_logodds_threshold=output_logodds_threshold)

        if solution.has_match():
            bm = solution.best_match()
            break
        else:
            logger.debug(f"{redf}: could not solve astrometry with {msg}.")

    ## If not, desist and return early; else we continue and build and save the wcs.

    if bm is None:
        return BuildWCSResult(success=False)
    else:
        logger.debug(f"{redf}: {msg} worked.")
        logger.debug(f"{redf}: {bm.index_path=}")
        logger.debug(f"{redf}: {bm.center_ra_deg=}")
        logger.debug(f"{redf}: {bm.center_dec_deg=}")
        logger.debug(f"{redf}: {bm.scale_arcsec_per_pixel=}")
        logger.debug(f"{redf}: {bm.logodds=}")
    
    # Build WCS from the best match

    wcs1 = WCS(bm.wcs_fields)

    if has_pairs:
        # Build WCS for pairs (just displace the center pixel by the disp_sign)
        wcs2 = wcs1.deepcopy()
        wcs2.wcs.crpix[0] += disp_sign[0]
        wcs2.wcs.crpix[1] += disp_sign[1]

    # save results and return

    return BuildWCSResult(success=True, 
                          wcslist=[wcs1, wcs2] if has_pairs else [wcs1], 
                          info=_save_astrocalib_proc_vars(locals()))




def _save_astrocalib_proc_vars(locals_dict):
    """ helper func to save all variables related to the source detection to a single dict. """

    astrocalib_proc_vars = dict()

    save_list = [
        'msg',
        'has_pairs',
        'bkg_box_size', 'bkg_filter_size',
        'bkg',
        'imgdata_bkg_substracted',
        'convolved_data',
        'seg_threshold', 'pos_seg', 'segment_map', 'seg_cat',
        'solution', 'bm', 'wcs1', 'header1',
        'stars', 'disp_sign',
    ]

    if locals_dict['has_pairs']:
        save_list += [
        'msg',
        'wcs2',
        'hist_range', 'bins', 'd_eps',
        'seg1', 'seg2', 'seg_d0', 'seg_disp_sign',
        'seg1_best', 'seg2_best', 'seg_disp_best', 'seg_disp_sign_best',
        'seg1xy', 'seg2xy', 'seg_disp_xy', 'seg_disp_sign_xy',
        'seg1xy_best', 'seg2xy_best', 'seg_disp_xy_best', 'seg_disp_sign_xy_best',
        ]

    astrocalib_proc_vars.update({k: v for k, v in locals_dict.items() if k in save_list})

    return astrocalib_proc_vars












# TODO: astrometry package .solve() leaks memory, which does not get freed even after gc.collect()
# and we ran many trials, until the leak builds up and crashesh the computer. As a workaround, this
# wrappers wraps the function solve_astrometry in such a way that it runs in a separate process,
# therefore all memory leaked is freed after the function ends. The wrapper allows us to use the function
# as if it was not wrapped, and it is transparent to the user. A better solution would be to fix the
# memory leak in the astrometry package.
#
# The wrapper starts the processes and waits for it to finish, and then returns the result.
#
#"""
# The wrapper uses the `multiprocess` package instead of `multiprocessing` because the latter does 
# not allow for daemon processes to have children, and solve_astrometry is invoked from a Pool of 
# processes when iop4conf.nthreads > 1, and Pool's processes are daemons. Another 
# alternative would have been to start the concurrent threads using Process instead of Pool 
# with daemon=False. 
#
# Using PoolExecutor in both ocassions (for concurrency and for leak containment) does allow
# to have two nested PoolExecutors (see below) but then the wrapper can not be a decorator 
# because multiprocessing can not pickle the function (dill can, which is why mutltiprocess
# works).

def _queue_wrapper(q, f, *args, **kwargs):
    q.put(f(*args, **kwargs))

def run_in_process(f):
    from functools import wraps
    import multiprocess
    @wraps(f)
    def wrapper(*args, **kwargs):
        ctx = multiprocess.context.ForkContext()
        q = ctx.Queue()
        p = ctx.Process(target=_queue_wrapper, args=(q, f) + args, kwargs=kwargs)
        p.start()
        # then get the result
        result = q.get()
        # then join
        p.join()
        return result

    return wrapper
#"""

"""
def solve_astrometry(*args, **kwargs):
    from functools import wraps
    import concurrent.futures, multiprocessing
    
    with concurrent.futures.ProcessPoolExecutor(mp_context=multiprocessing.get_context('fork')) as executor:
        future = executor.submit(_solve_astrometry, *args, **kwargs)
        result = future.result()
    return result

# invokes the astrometry solver with given stars positions and size and position hints
# def _solve_astrometry ... # if using the two nested pool excutors
"""

@run_in_process
def solve_astrometry(stars, size_hint=None, position_hint=None, output_logodds_threshold=21, sip=False, positional_noise_pixels=1):
    """ Actually run Astrometry solver with the given stars and position and size hints. """

    def logodds_callback(logodds_list: list[float]) -> astrometry.Action:
        """ Return early if there is a match with logodds greater than 90. """
        if np.amax(logodds_list) > 90: return astrometry.Action.STOP
        return astrometry.Action.CONTINUE
    
    solver = astrometry.Solver(
        astrometry.series_5200.index_files(
            cache_directory=iop4conf.astrometry_cache_path,
            scales={0,1,2,3,4}, ## 1,4 might be faster
        ) #+ astrometry.series_4200.index_files(
          #  cache_directory=iop4conf.astrometry_cache_path,
          #  scales={0,1,2,3,4},
        #)
    )

    if not sip: # default astrometry.net solver
        solution_parameters = astrometry.SolutionParameters(logodds_callback=logodds_callback, 
                                                            output_logodds_threshold=output_logodds_threshold, 
                                                            sip_order=0, tune_up_logodds_threshold=None,
                                                            positional_noise_pixels=positional_noise_pixels)
    else: # but for a telescope we do not expect sip to be important
        solution_parameters = astrometry.SolutionParameters(logodds_callback=logodds_callback, output_logodds_threshold=output_logodds_threshold)

    # TODO: this code was here to confirm that it was .solve() that was leaking memory.
    # It was confirmed it, and I am leaving it here for future debugging.
    #from iop4lib.utils import get_mem_parent_from_child, get_total_mem_from_child
    #child_mem_before_solve = get_mem_parent_from_child()
    #total_mem_before_solve = get_total_mem_from_child()

    solution = solver.solve(
        stars=stars,
        size_hint=size_hint,
        position_hint=position_hint,
        solution_parameters=solution_parameters,
    )

    #child_mem_after_solve = get_mem_parent_from_child()
    #total_mem_after_solve = get_total_mem_from_child()
    #logger.warning(f"Child memory usage before solve from pid {os.getpid()}: {child_mem_before_solve/1024**2:.0f} MB, after solve {child_mem_after_solve/1024**2:.0f} MB, difference {(child_mem_after_solve-child_mem_before_solve)/1024**2:.0f} MB")
    #logger.warning(f"Total memory usage before solve from pid {os.getpid()}: {total_mem_before_solve/1024**2:.0f} MB, after solve {total_mem_after_solve/1024**2:.0f} MB, difference {(total_mem_after_solve-total_mem_before_solve)/1024**2:.0f} MB")

    return solution