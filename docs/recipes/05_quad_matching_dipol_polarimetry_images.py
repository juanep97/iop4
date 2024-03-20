# %% [markdown]
# # Quad matching in DIPOL polarimetry images

# %%
# %run 01_notebook_configuration.py

# To reduce disk usage, only a central cut of of the full field of the DIPOL 
# camera is actually saved.

# %%
# polarimetry (cut)
from IPython.display import display
from iop4lib.db import RawFit, ReducedFit
from iop4lib.utils.plotting import imshow_w_sources
rf_pol = RawFit.by_fileloc(f"OSN-T090/2023-11-06/BLLAC_R_IAR-0760.fts")
display(rf_pol)
imshow_w_sources(rf_pol.mdata)

# %% [markdown]
# The position of the cut is defined in the header of the raw file

# %%
rf_pol.header['XORGSUBF'], rf_pol.header['YORGSUBF']

# %% [markdown]
# We can compare it with the full field image of the photometry field

# %%
# photometry filed (full field)
rf_phot = RawFit.by_fileloc(f"OSN-T090/2023-11-06/BLLac_IAR-0001R.fit")
display(rf_phot)
imshow_w_sources(rf_phot.mdata)

# %% [markdown]
# Let's build the photometry field first

# %% 
redf_phot = ReducedFit.create(rf_phot)
redf_phot.build_file()

# %% [markdown]

# As we can see, the astrometric calibration was done with the blind astrometry.net
# solver as before.

# Next, if we go to the polarimetry field, we see that there are not enough stars in
# the image to do the blind astrometric calibration.

# %% [markdown]
# The calibration of this file will use the previously calibrated photometry field
# and compare quads of stars in both images to find the transformation between
# them, as we can see in the log:

# %%
redf_pol = ReducedFit.create(rf_pol)
redf_pol.build_file()

# %% 
# !ls {redf_pol.filedpropdir}

from IPython.display import Image
Image(filename=f"{redf_pol.filedpropdir}/astrometry_matched_quads.png")

# %% [markdown]
# The image shows the quads of stars that were matched in both images, their
# distance in the hash space and the distance in the image space.

# This quad matching is necessary because the number of stars in the polarimetry
# field is not enough for the astrometry.net solver to work, plus the scale of the
# images is too small for the default index files. This quad matching needs at 
# least 4 (ideally a few more) stars in the image, and a previously calibrated 
# photometry field. If any of these conditions are not met, other methods must
# be used to calibrate the image (if there are only two bright sources at the 
# right distance, can be assumed that they are the target star, and if there are
# between 3 and 4-6 stars, we can try to match to known sources in the field).

# The process is implemented in the DIPOL._build_wcs_for_polarimetry_images_photo_quads
# method, which is called by the ReducedFit.astrometric_calibration method.
# We could also manually follow the process:

# %%
import numpy as np
import itertools
from iop4lib.instruments import DIPOL
from iop4lib.utils.astrometry import BuildWCSResult
from photutils.aperture import CircularAperture
from pathlib import Path
import matplotlib as mplt
import matplotlib.pyplot as plt

# get the subframe of the photometry field that corresponds to this polarimetry field, (approx)
x_start = redf_pol.rawfit.header['XORGSUBF']
y_start = redf_pol.rawfit.header['YORGSUBF']

x_end = x_start + redf_pol.rawfit.header['NAXIS1']
y_end = y_start + redf_pol.rawfit.header['NAXIS2']

idx = np.s_[y_start:y_end, x_start:x_end]

photdata_subframe = redf_phot.mdata[idx]

# find 10 brightest sources in each field

sets_L = list()

for redf, data in zip([redf_pol, redf_phot], [redf_pol.mdata, photdata_subframe]):

    positions = DIPOL._estimate_positions_from_segments(redf=redf, data=data, n_seg_threshold=1.5, npixels=32, centering=None, fwhm=1.0)
    positions = positions[:10]

    sets_L.append(positions)

fig = plt.figure(figsize=(12,6), dpi=iop4conf.mplt_default_dpi)
axs = fig.subplots(nrows=1, ncols=2)

for ax, data, positions in zip(axs, [redf_pol.mdata, photdata_subframe], sets_L):
    imshow_w_sources(data, pos1=positions, ax=ax)
    candidates_aps = CircularAperture(positions[:2], r=10.0)
    candidates_aps.plot(ax, color="b")
    for i, (x,y) in enumerate(positions):
        ax.text(x, y, f"{i}", color="orange", fontdict={"size":14, "weight":"bold"})#, verticalalignment="center", horizontalalignment="center") 
    ax.plot([data.shape[1]//2], [data.shape[0]//2], '+', color='y', markersize=10)
    
axs[0].set_title("Polarimetry field")
axs[1].set_title("Photometry field")
fig.suptitle("astrometry detected_sources")
plt.show()

# %%

# Build the quads for each field
quads_1 = np.array(list(itertools.combinations(sets_L[0], 4)))
quads_2 = np.array(list(itertools.combinations(sets_L[1], 4)))

from iop4lib.utils.quadmatching import hash_ish, distance, order, qorder_ish, find_linear_transformation
hash_func, qorder = hash_ish, qorder_ish

hashes_1 = np.array([hash_func(quad) for quad in quads_1])
hashes_2 = np.array([hash_func(quad) for quad in quads_2])

all_indices = np.array(list(itertools.product(range(len(quads_1)),range(len(quads_2)))))
all_distances = np.array([distance(hashes_1[i], hashes_2[j]) for i,j in all_indices])

idx = np.argsort(all_distances)
all_indices = all_indices[idx]
all_distances = all_distances[idx]

indices_selected = all_indices[:6]
colors = ["r", "g", "b", "c", "m", "y"]

# %%

figs, axs = zip(*[plt.subplots(figsize=(6,4), dpi=iop4conf.mplt_default_dpi) for _ in range(2)])

for (i,j), color in list(zip(indices_selected, colors)): 
    
    tij = find_linear_transformation(qorder(quads_1[i]), qorder(quads_2[j]))[1]

    for ax, fig, data, quad, positions in zip(axs, figs, [redf_pol.mdata, photdata_subframe], [quads_1[i], quads_2[j]], sets_L):
        imshow_w_sources(data, pos1=positions, ax=ax)
        x, y = np.array(order(quad)).T
        ax.fill(x, y, edgecolor='k', fill=True, facecolor=mplt.colors.to_rgba(color, alpha=0.2))

plt.show()