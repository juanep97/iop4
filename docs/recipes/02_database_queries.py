# %% [markdown]
# # Database queries

# Note: this notebooks makes reference to other example notebooks, make sure that the following files are available in the same directory as this notebook:

# %%
# !ls *.py

# %% [markdown]
# Configure IOP4 and logging (see examples)
# %%
# %run 01_notebook_configuration.py

# %% [markdown]
# In IOP4, observing sessions, FITS files and observational results are all 
# represented by a Python class, and each instance of them is a row in the 
# corresponding table in the database. IOP4 uses the Django ORM to interact with
# the database (DB) through these models, so the user does not need to worry about
# keeping the results in sync with the database, building SQL queries or 
# updating the DB schema after each change in any of these classes.

# IOP4 models are available under the iop4lib.db submodule. The following models
# are available:
# - Epoch: an observing session, represented by the telescope and date 
# (of the night). It provides methods for downloading and making the initial 
# classification of files from each observing session.
# - RawFit: a raw FITS file
# - MasterBias, MasterDark and MasterDark: calibration frames built from the 
# corresponding raw files for each epoch.
# - ReducedFit: a reduced FITS file, built from the corresponding raw file after 
# applying calibration frames, astronometric calibration, etc.
# - AperPhotResult: result of performing aperture photometry on an image. It is 
# used as an intermediate object to compute photo-polarimetry because of two 
# reasons: to re-use them and to easy debugging of the final results.
# - PhotoPolResult: final result of photo-polarimetry.

# Each of the models is linked through attributes to the rest of models.

# Database queries can be made through the `.objects` attribute of this models, 
# which allows for filtering, ordering, etc of these.

# Let's query the DB for the last observing session:

# %%
from iop4lib.enums import *
from iop4lib.db import *

epoch = Epoch.objects.order_by('-night').first()
epoch

# %% [markdown]
# In an IPython terminal, this will pretty-print a description of `epoch`.
# To query the first of the "science" (`LIGHT`) files, we could access them by

# %%
rf = epoch.rawfits.order_by('-juliandate').first()
rf

# %% [markdown]
# If the image was already reduced, you can get the corresponding reduced FITS 
# with

# %%
redf = rf.reduced
redf

# %% [markdown]
# If you wanted to get a preview of the file, you could plot the 2D data which
# is accesible through the `redf.mdata` attribute. Otherwise, you can use one of
# the utility function in `iop4lib.utils.plotting`.

# %%
from iop4lib.utils.plotting import imshow_w_sources
imshow_w_sources(redf.mdata)

# %% [markdown]
# The reduced fit will link back to the raw file through `redf.rawfit`. It also 
# links to the used calibration frames through `redf.masterbias`, 
# `redf.masterdark`, `redf.masterbias` and `redf.masterdark`, and to the night
# through `redf.epoch`. You can check all related fields for a given model in 
# its [reference](/iop4/docs/iop4lib.html) (fields of class ForeignKey, OneToOneField or 
# ManyToManyField).

# More complex queries can be made using `.filter()` `.exclude()`. These will 
# return a reduced Django Queryset. For example, to query (and count) all files 
# whose filename contains `BLLac` (case in-sensitive)

# %%
qs = ReducedFit.objects.filter(filename__icontains="BLLac")
qs.count()

# %% [markdown]
# The result is a Django QuerySet, which can be further filtered, indexed or 
# converted to a list. To exclude all DIPOL observations you would do:
qs = qs.exclude(instrument=INSTRUMENTS.DIPOL)
qs.count()

# If you wanted to access the second of the returned one you could do
qs[2]

# %% [markdown]
# Epochs (observing nights) are uniquely identified by the telescope and date 
# corresponding to the day of the start of the night (in the DB schema, there 
# is an _unique constaint_ on the `telescope` and `night` fields). Raw FITS 
# files are defined uniquely by their epoch and file name. The convenience 
# functions `Epoch.by_epochname` and `RawFit.by_fileloc" can be used to directly
# search the DB with this string

# %%
Epoch.by_epochname("OSN-T090/2023-11-06").pk == Epoch.objects.get(telescope="OSN-T090", night="2023-11-06").pk

# %% [markdown]
# and

# %%
RawFit.by_fileloc("OSN-T090/2023-11-06/BLLAC_R_IAR-0760.fts").pk == RawFit.objects.get(epoch__telescope="OSN-T090", epoch__night="2023-11-06", filename="BLLAC_R_IAR-0760.fts").pk

# %% [markdown]
# In the previous examples, the difference between `.filter` and `.get` is that 
# the former will return a QuerySet (iterable) which might also be empty, the 
# later will return the single instance matching the query, and raise an 
# exception if there is none or multiple.

# In addition, we have inadvertently used the `field__query` syntax to access 
# methods or related fields. At the beginning we used `filename__icontains`
# instead of simply `filename` to search for a case-insensitive string in the 
# field. And now we have used `epoch__telescope` to access the `telescope` field
# of the related Epoch objects, since RawFit contains simply a reference to 
# `Epoch`, instead of duplicated `telescope` and `night` fields.

# %% [markdown]
# For more information about querying the DB, see the [Django ORM documentation](https://docs.djangoproject.com/en/4.2/topics/db/queries/).