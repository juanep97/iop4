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
# it successfuly calibrates the image. When done through `build_file()`, if 
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
# image such as:

# %%
from iop4lib.db import RawFit
rf = RawFit.by_fileloc("CAHA-T220/2022-09-18/caf-20220918-23:01:33-sci-agui.fits")


