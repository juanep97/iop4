# %% [markdown]
# # Step-by-step reduction

# The most common way to use IOP4 is by launching the `iop4` command to 
# automatically reduce some nights. However, you might be interested in reducing
# some images step by step (when debugging problems, or implementing new 
# instruments). In this notebook we will do exactly that, which is more or less 
# what `iop4.py` does when launched from the command line.

# %%
# %run 01_notebook_configuration.py

# %% [markdown]
# ## Fetching the data from the telescope archive.

# We might be interested in downloading all files from one night, or just a single 
# night. In any case, you can directly do

# %%
# warning: this might clean flags and some other fields:
from iop4lib.db import Epoch

epoch = Epoch.create(epochname="OSN-T090/2022-09-18")
epoch

# %% [markdown]

# When an epoch is created and `auto_link_rawfits=True`, IOP4 first attempts to
# list the remote telescope archive, then it reads the local archive. It tries
# to download missing files. If it cannot list the remote, but finds the
# local folder for that epoch, it will just give a warning. If it fails at
# both, it will return an error. This behavior can be tuned with 
# `check_remote_list`, `force_redownload` and `fallback_to_local` arguments to
# `create()`. Keep in mind that this will create the corresponding RawFit 
# instances if necessary.

# If you only want to fetch or create a raw fit instance you can equivalently do
# it with 

# %%
from iop4lib.db import RawFit
rf = RawFit.create(fileloc="OSN-T090/2022-09-18/BLLac-0001R.fit")
rf

# %% [markdown]
# Since this file was already created by the previous Epoch.create command, it
# gives a DEBUG message that the entry for this file already exists and re-uses 
# it.

# In this examples, `epochname` and `fileloc` are simply shorthands for  
# specifying `telescope`,`night` and `telescope`,`night`,`filename`,  
# in a way that matches the archive directory structure. These params uniquely 
# identify the observing epoch and the raw fits file, respectively.

# %% [markdown]
# ## Creating calibration frames

# Next step on a normal calibration process would be creating the master frames 
# for the current night. This can be done with 

# %% 
epoch.build_master_biases()
epoch.build_master_darks()
epoch.build_master_flats()

# %% [markdown]
# This will use all existing biases, darks and flats in `epoch` to create all 
# calibration frames (for each filter, etc) using as many images as possible.

# After this, we can create the reduced images for all images in the night

# %% 
epoch.reduce()

# %% [markdown]
# or just for our image of interest

# %% 
from iop4lib.db import ReducedFit
redf = ReducedFit.create(rawfit=rf)
redf

# %% [markdown]
# This will automatically select the appropriate master calibration frames from 
# its night (if they exist) or from the closest night. Otherwise, you can 
# specify which files to use by passing it as an argument to the creation 
# function.

# By default, ReducedFit.create will not create the reduced FITS file unless
# you explicitly pass `auto_build=True`. If you didn't, you can do it with

# %% 
redf.build_file()

# %% [markdown]
# This will create the reduced FITS file by applying the calibration frames to 
# the raw image, and will give it an appropriate WCS. To check the result, with 
# the astrometry included, you can use the utility function 
# `plot_preview_astrometry`:

# %%
from iop4lib.utils.plotting import plot_preview_astrometry
plot_preview_astrometry(redf)

# %% [markdown]
# In the images, we can see the coordiante frames the positions of the sources 
# in the catalog that appear in the image.

# Since the image it is a photometry image, we can directly compute 
# the relative photometry results with

# %%
redf.compute_relative_photometry()

from iop4lib.db import PhotoPolResult
PhotoPolResult.objects.filter(reducedfits__in=[redf]).first()

# %% [markdown]
# These results can already be inspected in the [web interface](/iop4/explore/plot/).