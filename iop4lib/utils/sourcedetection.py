import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)  

import math
import numpy as np

from astropy.stats import (
    # SigmaClip,
    sigma_clipped_stats,
)

from astropy.convolution import (
    convolve,
    convolve_fft,
)

from photutils.detection import DAOStarFinder

from photutils.background import (
    Background2D,
    # MedianBackground,
    SExtractorBackground,
)

from photutils.segmentation import (
    detect_sources,
    SourceFinder,
    SourceCatalog,
    make_2dgaussian_kernel,
)

from photutils.aperture import CircularAperture

from iop4lib.utils import next_odd

def get_bkg(imgdata, box_size=(16,16), filter_size=(11,11), mask=None, **bkg2d_kwargs):
    """Returns the 2D background for a given box_size and filter_size. Optionally, a mask can be provided (to mask sources or bad pixels)."""

    #bkg_estimator = MedianBackground()
    #bkg_estimator = MedianBackground(SigmaClip(sigma=3))

    bkg_estimator = SExtractorBackground() # default is to perform sigma clip with 3sigma and 5 iter
    #bkg_estimator = SExtractorBackground(SigmaClip(sigma=3.0))

    bkg = Background2D(imgdata, box_size, filter_size=filter_size, bkg_estimator=bkg_estimator, mask=mask, **bkg2d_kwargs)

    return bkg

def apply_gaussian_smooth(data, fwhm, kernel_size=None):
    if kernel_size is None:
        kernel_size = 2*int(fwhm)+1

    if kernel_size < 30:
        fconv = convolve
    else:
        fconv = convolve_fft

    kernel = make_2dgaussian_kernel(fwhm, size=kernel_size)
    data = fconv(data, kernel)
    return data

def get_sources_daofind(data, threshold=None, fwhm=8.0, n_threshold=5.0, brightest=100, exclude_border=True):
    """
    `data` needs not but should be bkg-substracted and smoothed (that is, convolved with a kernel),
    threshold should be set depending on the brackground noise.
    """

    mean, median, std = sigma_clipped_stats(data, sigma=5.0)

    if threshold is None:
        if n_threshold is not None:
            threshold = 5.0*std
        else:
            raise ValueError('threshold or n_threshold must be provided')

    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, brightest=brightest, exclude_border=exclude_border)  
    sources = daofind(data-median)
    
    if sources is None or len(sources) == 0:
        return np.empty((0,0))
    
    sources.sort('flux', reverse=True)

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    
    return sources, positions

def get_cat_sources_from_segment_map(segment_map, imgdata_bkg_substracted, convolved_data):    
    """
    From a segmentation map, return a cataloge and the positions ((x1,y1), (x2,y2), ...).
    
    Needs also the bkg-substracted image and the convolved map to find the centroids from the segement map.
    """
    cat = SourceCatalog(data=imgdata_bkg_substracted, segment_img=segment_map, convolved_data=convolved_data)
    tb = cat.to_table()
    tb.sort('segment_flux', reverse=True)
    pos = np.transpose((tb['xcentroid'], tb['ycentroid']))
    return cat, pos, tb


def get_segmentation(imgdata_bkg_substracted, threshold, fwhm=1.0, kernel_size=None, npixels=64, deblend=True, mask=None):
    """
    Given the image with the background substracted, convolve it with a kernel (smooth it) and 
    find the segmentation map.
    
    Return the segemented map and the convolved image.
    """
    
    if kernel_size is None:
        kernel_size = 2*int(fwhm)+1

    if kernel_size < 30:
        fconv = convolve
    else:
        fconv = convolve_fft

    kernel = make_2dgaussian_kernel(fwhm, size=kernel_size)
    convolved_data = fconv(imgdata_bkg_substracted, kernel)

    if deblend:
        finder = SourceFinder(npixels=npixels, deblend=deblend, progress_bar=False)
        segment_map = finder(convolved_data, threshold)
    else:
        segment_map = detect_sources(convolved_data, threshold, npixels=npixels)

    if mask is not None:
        segment_map.remove_masked_labels(mask)

    return segment_map, convolved_data

def mask_other_sources_from_centroids(data, r, fwhm, exclude=None):
    
    # bkg_box_size = next_odd(10*fwhm)
    bkg_box_size = data.shape
    bkg = get_bkg(data, filter_size=3, box_size=bkg_box_size, exclude_percentile=90)
    data = data - bkg.background
        
    n_seg_threshold = 3
    npixels = max(9, math.ceil(fwhm**2))
    
    seg_threshold = n_seg_threshold * bkg.background_rms
    segment_map, convolved_data = get_segmentation(data, fwhm=fwhm, npixels=npixels, threshold=seg_threshold)

    if segment_map is None:
        return np.zeros_like(data, dtype=bool), []

    seg_cat, positions, tb = get_cat_sources_from_segment_map(segment_map, data, convolved_data)

    total_mask = np.full(data.shape, False)

    if exclude:
        exclude_mask = np.full(data.shape, False)
        for pos in exclude:
            mask = CircularAperture(pos, r=r).to_mask().to_image(data.shape).astype(bool)
            exclude_mask = exclude_mask | mask

    final_positions = list()
    
    for pos in positions:
        
        mask = CircularAperture(pos, r=r).to_mask().to_image(data.shape).astype(bool)

        if exclude:
            overlap_frac = np.sum(exclude_mask & mask) / np.sum(mask)
            if overlap_frac > 0.5:
                continue
        
        total_mask = total_mask | mask
        
        final_positions.append(pos)

    return total_mask, final_positions

def mask_other_sources_from_segmap(data, r, fwhm, exclude=None):
    
    # bkg_box_size = next_odd(10*fwhm)
    bkg_box_size = data.shape
    bkg = get_bkg(data, filter_size=3, box_size=bkg_box_size, exclude_percentile=90)
    data = data - bkg.background
    
    n_seg_threshold = 3
    npixels = max(9, math.ceil(fwhm**2))

    seg_threshold = n_seg_threshold * bkg.background_rms
    segment_map, convolved_data = get_segmentation(data, fwhm=fwhm, npixels=npixels, threshold=seg_threshold)

    if segment_map is None:
        return np.zeros_like(data, dtype=bool), []

    exclude_mask = None

    if exclude:
        exclude_mask = np.zeros(data.shape, dtype=bool)

        for pos in exclude:
            mask = CircularAperture(pos, r=r).to_mask().to_image(data.shape).astype(bool)
            exclude_mask |= mask

        segment_map.remove_masked_labels(exclude_mask)

    total_mask = segment_map.make_source_mask(size=next_odd(r))

    seg_cat, positions, tb = get_cat_sources_from_segment_map(segment_map, data, convolved_data)

    return total_mask, positions

mask_other_sources = mask_other_sources_from_centroids
# mask_other_sources = mask_other_sources_from_segmap
