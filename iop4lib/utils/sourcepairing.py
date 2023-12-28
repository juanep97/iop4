import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)  

import numpy as np
import scipy as sp

import matplotlib as mplt
import matplotlib.pyplot as plt

import itertools
from scipy.spatial import cKDTree



def get_pairs_d(pos, d0=None,
                d_eps=None, d_min=None, d_max=None,
                bins=None, hist_range=None, redf=None, doplot=False, ax=None):
    """
    From a list of positions, finds the most common distance between them (d0),
    and pairs the points that are at such distance. If d0 is given, it is used instead of computing it.
    
    The pairs are ordered such that for pair (p1, p2), p1 is always to the left (smaller x value) than p2.
    """

    d_eps = d_eps or 0.8
    d_min = d_min or 0
    d_max = d_max or 60

    if bins is None:
        if redf is not None:
            bins = int( 0.75 * max(redf.data.shape) )
        else:
            raise ValueError("bins must be specified if redf is not given")

    if hist_range is None:
        if redf is not None:
            hist_range = (0, min(redf.data.shape))
        else:
            raise ValueError("hist_range must be specified if redf is not given")
        
    if pos is None or len(pos) < 2:
        return [], [], None, None

    pairs = list(itertools.combinations(pos, 2))
    distances = [np.linalg.norm(p1-p2) for p1,p2 in pairs]

    if d0 is None:
        hist, edges = np.histogram(distances, bins=bins, range=hist_range)
        centers = (edges[:-1]+edges[1:])/2

        idx = (d_min <= centers) & (centers <= d_max)
        idx_max = np.argmax(hist[idx])
        d0 = centers[idx][idx_max]

    paired = [(p1,p2) for p1,p2 in pairs if np.abs(np.linalg.norm(p1-p2)-d0) < d_eps]
    paired = [[p1,p2]  if p1[0]>p2[0] else [p2,p1] for (p1,p2) in paired]

    if len(paired) == 0:
        return [], [], d0, None

    list1, list2 = list(zip(*paired))
    
    pos1 = np.array(list1)
    pos2 = np.array(list2)
    disp_sign = np.mean(pos2-pos1, axis=0)

    # Plotting (optional)
    if doplot:
        if ax is None:
            ax = plt.gca()
        cnts, edges, bars = ax.hist(distances,  bins=bins, range=hist_range)
        ax.axvline(x=d0, color='r', linestyle='--', linewidth=1, alpha=0.5)
        bars[np.argmax(cnts)].set_facecolor('red')

    return list1, list2, d0, disp_sign



def get_pairs_dxy(pos, disp=None, 
                  dx_eps=None, dy_eps=None, d_eps=None, dx_min=None, dx_max=None, dy_min=None, dy_max=None, d_min=None,
                  bins=None, hist_range=None, redf=None, doplot=False, axs=None, fig=None):
    """
    From a list of positions, finds the most common distances between them in both x and y axes (disp),
    and pairs the points that are at such distances.

    If disp is given, it is used as the most common distance in both axes instead of computing it.
    
    The pairs are ordered such that for pair (p1, p2), p1 is always to the left (smaller x value) than p2.
    
    Note: this function is similar to get_pairs_d(), but finds the most common distances both in x and y axes.
    """
    
    dx_eps = dx_eps or 0.8
    dy_eps = dy_eps or 0.8
    d_eps = d_eps or 0.8
    dx_min = dx_min or 0
    dx_max = dx_max or 60
    dy_min = dy_min or 0
    dy_max = dy_max or 60
    d_min = d_min or 0

        
    if pos is None or len(pos) < 2:
        return [], [], None, None
    
    pairs = list(itertools.combinations(pos, 2))

    if disp is None:

        if bins is None:
            if redf is not None:
                bins = int( 0.75 * max(redf.data.shape) )
            else:
                raise ValueError("bins must be specified if redf is not given")

        if hist_range is None:
            if redf is not None:
                hist_range = (0, min(redf.data.shape))
            else:
                raise ValueError("hist_range must be specified if redf is not given")
        
        disp = list()
        for i, d_min, d_max in zip([0, 1], [dx_min, dy_min], [dx_max, dy_max]): # for each axis
            distances = [abs(p1[i]-p2[i]) for p1,p2 in pairs]

            hist, edges = np.histogram(distances, bins=bins, range=hist_range)
            centers = (edges[:-1]+edges[1:])/2

            idx = (d_min <= centers) & (centers <= d_max)
            idx_max = np.argmax(hist[idx])
            d0 = centers[idx][idx_max]

            disp.append(d0)

    paired = [(p1,p2) for p1,p2 in pairs if ( abs( abs( p1[0] - p2[0] ) - disp[0] ) < dx_eps and abs( (abs( p1[1] - p2[1] ) - disp[1] ) ) < dy_eps )]
    paired = [[p1,p2]  if p1[0]>p2[0] else [p2,p1] for (p1,p2) in paired]

    if len(paired) == 0:
        return [], [], disp, None
    
    list1, list2 = list(zip(*paired))
    
    pos1 = np.array(list1)
    pos2 = np.array(list2)
    disp_sign = np.mean(pos2-pos1, axis=0)

    # Plotting (optional)
    if doplot:
        if axs is None:
            if fig is None:
                fig = plt.gcf()
            axs = fig.subplots(nrows=2, sharex=True)

        if len(axs) == 1:
            ax = axs[0]
            for i, color in zip([0, 1], ['r','b']):
                distances = [abs(p1[i]-p2[i]) for p1,p2 in pairs]
                cnts, edges, bars = ax.hist(distances, bins=bins, range=hist_range, alpha=0.3)
                ax.axvline(x=disp[i], color=color, linestyle='--', linewidth=1, alpha=0.3)
                bars[np.argmax(cnts)].set_facecolor('red')
        elif len(axs) == 2:
            for i in [0, 1]:
                distances = [abs(p1[i]-p2[i]) for p1,p2 in pairs]
                cnts, edges, bars = axs[i].hist(distances, bins=bins, range=hist_range)
                axs[i].axvline(x=disp[i], color='k', linestyle='--', linewidth=1, alpha=0.5)
                bars[np.argmax(cnts)].set_facecolor('red')
        else:
            raise ValueError("axs must be a list of length 1 or 2")

    return list1, list2, disp, disp_sign



def get_best_pairs(list1, list2, disp_sign, dist_err=None, disp_sign_err=None):
    """
    From two lists which correspond to paired points, if there are points participating in more
    than one pair, return only the best pair according to displacement disp_sign. 
    
    If dist_err (scalar) or disp_sign_err are given, only pairs whose points are displace by disp_sign within 
    a distance of dist_err (1d) or within disp_sign_err (2d) are considered.

    Example
    -------
    An example of pair finding where first we detect the sources and find the pairs without a priori knowledge of the displacement.
    ```
    # Detect sources in the sky (pos_seg is a list of (x,y) positions)

    # Define some parameters
    
    d_eps = 0.8
    bins = int( 0.75 * max(redf.data.shape) )
    hist_range = (0, min(redf.data.shape))

    ## Find pairs of sources matching in scalar distance, using the most common distance between sources
    seg1, seg2, seg_d0, seg_disp_sign = get_pairs_d(pos_seg, d_eps=d_eps, bins=bins, hist_range=hist_range)

    ## Get only the best pairs, according to the displacement sign
    seg1_best, seg2_best, seg_disp_best, seg_disp_sign_best = get_best_pairs(seg1, seg2, seg_disp_sign)

    ## Find pairs of sources matching in 2d position, using the most common displacement between sources
    seg1xy, seg2xy, seg_disp_xy, seg_disp_sign_xy = get_pairs_dxy(pos_seg, d_eps=d_eps, bins=bins, hist_range=hist_range)

    ## Get only the best pairs, according to the displacement sign
    seg1xy_best, seg2xy_best, seg_disp_xy_best, seg_disp_sign_xy_best = get_best_pairs(seg1xy, seg2xy, seg_disp_sign_xy)
    ```
    Alternatively, using a priori knowledge of the displacement, we can find the pairs directly:
    ```
    # Get the average displacement between pairs and its std from already calibrated sources
    disp_mean = np.mean([redf.astrometry_info[-1]['seg_disp_sign_xy'] for redf in ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.BUILT_REDUCED, obsmode="POLARIMETRY") if 'seg_disp_sign_xy' in redf.astrometry_info[-1] and isinstance(redf.astrometry_info[-1]['seg_disp_sign_xy'], np.ndarray)], axis=0)
    disp_std = np.std([redf.astrometry_info[-1]['seg_disp_sign_xy'] for redf in ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.BUILT_REDUCED, obsmode="POLARIMETRY") if 'seg_disp_sign_xy' in redf.astrometry_info[-1] and isinstance(redf.astrometry_info[-1]['seg_disp_sign_xy'], np.ndarray)], axis=0)

    # Detect sources (pos_seg is a list of (x,y) positions)

    # Directly find the best pairs with the mean disp_mmmean and disp_std
    seg1xy_best, seg2xy_best, seg_disp_xy_best, seg_disp_sign_xy_best = get_best_pairs(pos_seg, pos_seg, disp_mean, disp_sign_err=5*disp_std)
    ```

    
    Parameters
    ----------
    list1, list2 : list, list
        paired lists of points
    disp_sign: (float, float)
        the displacement between points
    
    Returns
    -------
    list1, list2 : list, list
        paired list of points
    d0_new : float
        recomputed points
    disp_sign_new : 
        recomputed displacement
    """
    
    if list1 is None or len(list1) < 2:
        return [], [], None, None

    set1 = {tuple(p1) for p1 in list1}
    set2 = {tuple(p2) for p2 in list2}

    def get_best_companion(p,pL):
        p = np.array(p) ## so now p+disp_sign-x is an array even if disp_sign and x are tuples
        return min(pL, key=lambda x: np.abs(np.linalg.norm(p+disp_sign-x)))

    paired = [(p1, get_best_companion(p1,set2)) for p1 in set1]

    if dist_err is not None:
        paired = [pair for pair in paired if np.linalg.norm(pair[0]+disp_sign-pair[1]) < dist_err]
    
    if disp_sign_err is not None:
        paired = [pair for pair in paired if np.abs(pair[0][0]+disp_sign[0]-pair[1][0]) < disp_sign_err[0]]
        paired = [pair for pair in paired if np.abs(pair[0][1]+disp_sign[1]-pair[1][1]) < disp_sign_err[1]]

    if len(paired) == 0:
        return [], [], None, None

    list1, list2 = list(zip(*paired))
    
    pos1 = np.array(list1)
    pos2 = np.array(list2)
    disp_sign_new = np.mean(pos2-pos1, axis=0)
    d0_new = np.linalg.norm(disp_sign_new)
    
    return list1, list2, d0_new, disp_sign_new

