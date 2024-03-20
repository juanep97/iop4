# %% [markdown]
# # Notebook configuration
# Before using IOP4 inside a jupyter notebook, you need to configure the DB with the following lines

# %%
# %autoawait off
import iop4lib.config
iop4conf = iop4lib.Config(config_db=True, gonogui=False, jupytermode=True) 

# %% [markdown]
# Now that IOP4 is configured, you can import and use the DB models. E.g.:
# ```ipython
# from iop4lib.db import RawFit, ReducedFit
# ```

# %% [markdown]
# If you want to use the autoreload extension, e.g. with
# ```ipython
# %load_ext autoreload
# %autoreload all
# ```
# at the top of your notebook. You might encounter some problems when the modifications affect the DB models. In that case, re-executing the import statement a few times should help, otherwise you will need to restart the kernel.

# Since you are not using the `iop4.py` script, you will need to configure logging to suit your needs. E.g, to log to standard output, you can use:

# %%
import sys

# configure logging
import coloredlogs
import logging

logger = logging.getLogger()

# remove handlers

for handler in logger.handlers:
    logger.removeHandler(handler)

# set debug level and a nicer format for notebooks

logger.setLevel(logging.DEBUG)

logger_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(logger_handler)

iop4conf.log_date_format = '%Y-%m-%d %H:%M'
iop4conf.log_format = '%(asctime)s - pid %(process)d - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s'

logger_handler.setFormatter(coloredlogs.ColoredFormatter(iop4conf.log_format, datefmt=iop4conf.log_date_format))

# %% [markdown]
# Now you are ready to start using IOP4 inside your notebook.