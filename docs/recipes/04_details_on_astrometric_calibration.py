# %% [markdown]
# # Details on astrometric calibration

# %%
# %run 01_notebook_configuration.py

# %% [markdown]
# The header of raw FITS files does not usually contain a valid WCS. The 
# astrometric calibration is implemented automatically during the 
# `ReducedFit.build_file()` step, and can be done separately through 
# `ReducedFit.astrometric_calibration()`.

# %%
from iop4lib.db import ReducedFit
redf = ReducedFit.by_fileloc("OSN-T090/2022-09-18/BLLac-0001R.fit")
redf.build_file()

# %% [markdown]
# For simple photometry images such as those from the Andor cameras, tries to
# detect sources in the image with different sets of parameters and 
# feeds them to a local astrometry.net solver. It tries with several sets until 
# it successfully calibrates the image. When done through `build_file()`, if 
# calibration is not achieved, it will give the ERROR_ASTROMETRY flag, 
# otherwise it will give it the BUILT_REDUCED flag.

# %%
redf

# %% 
redf.flag_labels

# %% [markdown]
# In the reduction log
# ```
# 2024-02-21 11:51 - pid 75152 - [plotting.py:476] - DEBUG - <ReducedFit 212 | OSN-T090/2022-09-18/BLLac-0001R.fit>: plotting astrometry summary image of background substraction results
# 2024-02-21 11:51 - pid 75152 - [plotting.py:488] - DEBUG - <ReducedFit 212 | OSN-T090/2022-09-18/BLLac-0001R.fit>: plotting astrometry summary image of segmentation results
# 2024-02-21 11:51 - pid 75152 - [plotting.py:582] - DEBUG - <ReducedFit 212 | OSN-T090/2022-09-18/BLLac-0001R.fit>: plotting astrometry summary image of astrometry results
# ``` 

# %% tags=["remove_input"]
from IPython.display import Markdown as md
md(f"you can see that several summary images were created. They are displayed in the [admin page for the reduced fit](/iop4/admin/iop4api/reducedfit/details/{redf.pk})")

# %% [markdown]
# For images with ordinary and extraordinary pairs such as those from CAFOS and 
# DIPOL, the reduction procedure first finds the sources and then divides them 
# in ordinaty and extraordinary lists. This is done through the utilities at
# `iop4lib.utils.sourcepairing` submodule.

# To find the pairs, the algorithm builds a distribution of the distance between
# all sources in the image, and selectes the most frequent distance with some
# resolution. This process is not free from error. It can be improved by 
# providing some bounds for this expected distance. For example, in a CAFOS 
# image such as

# %%
from iop4lib.db import RawFit
rf = RawFit.by_fileloc("CAHA-T220/2022-09-18/caf-20220918-23:01:33-sci-agui.fits")

# %% [markdown]
# We first create the reduced image and apply the calibration frames

# %%
redf = ReducedFit.create(rawfit=rf)
redf.apply_masters()

# %% [markdown]
# The `.has_pairs` property tells whether an image is expected to have pairs:

# %%
redf.has_pairs

# %% [markdown]
# Since we will be doing "blind" astrometric calibration with the local 
# astrometry.net solver, we will need (or at least greatly benefit) from hints
# for the size of the pixel and for pointing of the camera. These can be 
# obtained with 

# %%
size_hint = redf.get_astrometry_size_hint()
size_hint

# %% [markdown]
# This code to build this size hint is implemented in the corresponding 
# instrument class.

# %% [markdown]
# We will also benefit from a hint for the position of the object in the image.

# %% [markdown]
# The position is hint is built by the corresponding instrument class by looking
# at the RA and DEC (or equivalent) keywords in the FITS header.
# If the FITS header contains the RA and DEC (or equivalent) keywords, the 
# `header_hintcoord` property will return an SkyCoord object with the position

# %%
redf.header_hintcoord

# %% [markdown]

# This position should be close to the observed object. 
# The header_hintobject property returns the AstroSource in the catalog
# that matches the name in the OBJECT (or equivalent) keyword:

# %%
redf.header_hintobject

# %%
redf.header_hintobject.coord.separation(redf.header_hintcoord)

# %% [markdown]
# It can be be seen that the separation is indeed small.

# When header_hintcoord is available, 
# get_astrometry_position_hint() will return a hint for the object built from 
# it. Otherwise, it will look at the header_hintobject. If none are available, 
# it will raise an exception.

# %%
position_hint = redf.get_astrometry_position_hint()
position_hint

# %% [markdown]

# All these hints will be used to accelerate the blind calibration.

# Now, let's detect the stars visible in the image. The whole process will 
# depend on the parameters we select, and we might need to try with several 
# combinations. 

# %%
bkg_filter_size = 11 # a higher number will produce a more smooth bkg
bkg_box_size = 16 # a higher number will produce a more detailed bkg
# these bkg parameters should be set so the computed bkg does not include light from the point sources, but is detailed enough to show real variations along the image.
seg_kernel_size = None # none will automatically build a kernel of size 2n+1, which should be almost always a good approximation
npixels = 32 # related to the size of the sources in the image
seg_fwhm = 1.0 # the image will be convolved with a gaussian kernel of this width before src detection
n_rms_seg = 1.0 # the threshold of signal over background to count as a detection
# these detection parameters should return most of the real points in the image without too much fake sources
keep_n_seg = 200 # the maximum number of detected sources to use in the calibration
border_margin_px = 20 # remove any pixels less than this distance from the borders

# %% [markdown]
# We can use the utility functions in 
# `iop4lib.utils.sourcedetection` to get an bkg-estimation, which utilize the 
# `photutils` package (you could do the same, or try different methods following
# [its docs](https://photutils.readthedocs.io/en/stable/index.html).

# ### Background substraction

# %%
from iop4lib.utils.sourcedetection import get_bkg, get_segmentation, get_cat_sources_from_segment_map
bkg = get_bkg(redf.mdata, filter_size=bkg_filter_size, box_size=bkg_box_size)
imgdata_bkg_substracted = redf.mdata - bkg.background

from iop4lib.utils.plotting import plot_preview_background_substraction_1row
import matplotlib.pyplot as plt
import matplotlib as mplt
fig = plt.figure(figsize=(12, 5))
plot_preview_background_substraction_1row(redf, bkg, fig=fig)
plt.show()

# %% [markdown]
# ### Image Segmentation

# %%
seg_threshold = n_rms_seg * bkg.background_rms
segment_map, convolved_data = get_segmentation(imgdata_bkg_substracted, fwhm=seg_fwhm, kernel_size=seg_kernel_size, npixels=npixels, threshold=seg_threshold)
seg_cat, pos_seg, tb = get_cat_sources_from_segment_map(segment_map, imgdata_bkg_substracted, convolved_data)

# %%
from astropy.visualization import SqrtStretch, LogStretch, AsymmetricPercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize
fig, ax = plt.subplots()
ax.imshow(imgdata_bkg_substracted, cmap='gray', norm=ImageNormalize(imgdata_bkg_substracted.compressed(), stretch=LogStretch(a=10), interval=AsymmetricPercentileInterval(30, 99)))
seg_cat.plot_kron_apertures(color='r', lw=1.5)
plt.show()

# %% [markdown]
# ### Source pairing
# Now we need to distinguish between ordinary and extraordinary sources. To facilitate 
# this, you can use the `get_pairs_dxy` function, which will try to separate 
# them by looking at the most common distance between sources.

# First we find the most common distance between sources (both in x and y)

# %% 
from iop4lib.utils.sourcepairing import get_pairs_dxy, get_best_pairs
bins = 60
hist_range = (0, 60)
dx_min = dy_min = 0
dx_max = dy_max = 60
dx_eps = dy_eps = 1.0
seg1xy, seg2xy, seg_disp_xy, seg_disp_sign_xy = get_pairs_dxy(pos_seg, bins=bins, hist_range=hist_range, dx_min=dx_min, dx_max=dx_max, dy_min=dy_min, dy_max=dy_max, dx_eps=dx_eps, dy_eps=dy_eps, doplot=True)
seg_disp_sign_xy

# %% [markdown]
# Then we select the best pairs according to this distance, and recalculate it

# %%
seg1xy_best, seg2xy_best, seg_disp_xy_best, seg_disp_sign_xy_best = get_best_pairs(seg1xy, seg2xy, seg_disp_sign_xy)
seg_disp_sign_xy_best

# %% [markdown]
# We can plot the image again and check that indeed the pairs are well separated

# %%
from iop4lib.utils.plotting import imshow_w_sources
imshow_w_sources(redf.mdata, pos1=seg1xy_best, pos2=seg2xy_best)

# %% [markdown]
# To avoid mistakes, we provided lower and upper bounds for the distance between
# pairs. This can be specially important if there are few sources in the image.
# We could also do the same by ourselves, by looking at the distribution of
# distances:

# %%
import itertools
pairs = list(itertools.combinations(pos_seg, 2))
len(pairs)

# %%
dx_min = 0
dx_max = 60
dy_min = 0
dy_max = 60
paired = [(p1,p2) for p1,p2 in pairs if ( dx_min < abs( p1[0] - p2[0] ) < dx_max and dy_min < abs( p1[1] - p2[1] ) < dy_max )]
len(paired)

# %%
fig, ax = plt.subplots()
for i in [0,1]:
    distances = [abs(p1[i]-p2[i]) for p1,p2 in paired]
    plt.hist(distances, bins=60, range=(0,60), alpha=0.7)
plt.xlabel('Distance (px)')
plt.ylabel('Number of pairs')
plt.show()

# %% [markdown]
# Finally, we need to invoke the local astrometry.net solver to calibrate the image.

# %%
from iop4lib.utils.astrometry import solve_astrometry

solution = solve_astrometry(seg1xy_best, size_hint=size_hint, position_hint=position_hint)
solution

# %% [markdown]
# We can see that the solver was successful, and that the result has a logodds much
# higher than the default minimum of 21, meaning that the calibration was easy and trustable.

# The WCS (for one set of pairs) can be obtained with

# %%
from astropy.wcs import WCS
wcs1 = WCS(solution.best_match().wcs_fields)

# %% [markdown]
# We can plot the image with the sources and the WCS overlayed, and the catalog
# sources overplotted

# %%

fig, ax = plt.subplots(subplot_kw={'projection': wcs1})

imshow_w_sources(redf.mdata, ax=ax)

from iop4lib.db import AstroSource
from photutils.aperture import CircularAperture

for src in AstroSource.get_sources_in_field(wcs1, width=redf.mdata.shape[1], height=redf.mdata.shape[0]):
    ax.plot(*src.coord.to_pixel(wcs1), 'rx')
    ax.annotate(src.name, src.coord.to_pixel(wcs1), xytext=(20,0), textcoords='offset points', color='red', weight='bold')

ax.coords.grid(color='white', linestyle='solid')

ax.coords['ra'].set_axislabel('Right Ascension')
ax.coords['ra'].set_ticklabel_visible(True)
ax.coords['ra'].set_ticklabel_position('lb')

ax.coords['dec'].set_axislabel('Declination')
ax.coords['dec'].set_ticklabel_visible(True)
ax.coords['dec'].set_ticklabel_position('lb')

plt.show()

# %% [markdown]
# We can see that the sources are well aligned with the catalog sources. Since we
# already know the separation between the pairs, we can directly build the other 
# wcs by displacing the first one. However this will be done automatically when 
# we call the higher level methods in the `ReducedFit` class. In fact, the 
# `ReducedFit.astrometric_calibration()` method will do all the above and save 
# both WCS in the FITS header of the ReducedFit object:

# %%
redf.astrometric_calibration()

# %% 
redf.wcs1

# %%
redf.wcs2

# %% 
fig, ax = plt.subplots(subplot_kw={'projection': wcs1})

imshow_w_sources(redf.mdata, ax=ax)

from iop4lib.db import AstroSource
from photutils.aperture import CircularAperture

for src in AstroSource.get_sources_in_field(wcs1, width=redf.mdata.shape[1], height=redf.mdata.shape[0]):
    CircularAperture([*src.coord.to_pixel(redf.wcs1)], r=20).plot(ax=ax, color='r')
    ax.annotate(src.name, src.coord.to_pixel(redf.wcs1), xytext=(20,0), textcoords='offset points', color='red', weight='bold', verticalalignment='center')
    ax.plot(*src.coord.to_pixel(redf.wcs2), 'rx')

ax.coords.grid(color='white', linestyle='solid')

ax.coords['ra'].set_axislabel('Right Ascension')
ax.coords['ra'].set_ticklabel_visible(True)
ax.coords['ra'].set_ticklabel_position('bl')

ax.coords['dec'].set_axislabel('Declination')
ax.coords['dec'].set_ticklabel_visible(True)
ax.coords['dec'].set_ticklabel_position('lb')
#ax.coords['dec'].set_ticklabel(rotation='vertical')

plt.show()

# %% [markdown]
# If the calibration was successful, summary images containing more or less the 
# same information as the ones we plotted will be created and should be displayed 
# in the admin page for the reduced fit:

# %% tags=["remove_input"]
from IPython.display import Markdown as md
md(f"[/iop4/admin/iop4api/reducedfit/details/{redf.pk}](/iop4/admin/iop4api/reducedfit/details/{redf.pk})")