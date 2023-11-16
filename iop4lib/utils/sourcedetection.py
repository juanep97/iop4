import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)  

import numpy as np

from photutils.detection import DAOStarFinder
from astropy.convolution import convolve, convolve_fft
from photutils.segmentation import make_2dgaussian_kernel
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceFinder
from photutils.segmentation import SourceCatalog


def get_bkg(imgdata, box_size=(16,16), filter_size=(11,11), mask=None):
    """Returns the 2D background for a given box_size and filter_size. Optionally, a mask can be provided (to mask sources or bad pixels)."""

    #bkg_estimator = MedianBackground()
    #bkg_estimator = MedianBackground(SigmaClip(sigma=3))

    bkg_estimator = SExtractorBackground() # default is to perform sigma clip with 3sigma and 5 iter
    #bkg_estimator = SExtractorBackground(SigmaClip(sigma=3.0))

    bkg = Background2D(imgdata, box_size, filter_size=filter_size, bkg_estimator=bkg_estimator, mask=mask)

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










# other functions

from sklearn.cluster import KMeans

def select_points(points, n, brightness=None):
    """ returns n points approximately uniformly distributed over the image, selecting the closes one to each cluster or the brightest one if brightness is provided."""

    # Convert the list of tuples to a 2D numpy array
    points_arr = np.array(points)

    # Create a KMeans instance
    kmeans = KMeans(n_clusters=n, random_state=0, n_init=10)

    # Perform the clustering
    kmeans.fit(points_arr)

    # Get labels for all points
    labels = kmeans.labels_

    selected_points = []
    selected_points_idx = []
    clusters = []
    for i in range(n):
        # Get the points in this cluster
        cluster_points = points_arr[labels == i]

        if brightness is not None:
            # Get the brightness values for these points
            cluster_brightness = np.array(brightness)[labels == i]

            # Find the index of the brightest point
            brightest_idx = np.argmax(cluster_brightness)
        else:
            brightest_idx = np.argmin(np.linalg.norm(points_arr[labels == i]-kmeans.cluster_centers_[:, np.newaxis]))

        # Add the brightest point to the list
        selected_points.append(cluster_points[brightest_idx])
        ## add also the index 
        selected_points_idx.append(np.argwhere(labels == i)[brightest_idx][0])

        # Append the cluster points to the clusters list
        clusters.append(cluster_points.tolist())

    # Return the brightest points and the clusters
    return selected_points, selected_points_idx, clusters, kmeans