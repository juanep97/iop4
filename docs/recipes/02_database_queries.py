# %% [markdown]
# # Database queries

# ```Note: this notebooks makes reference to other example notebooks, make sure that the following files are available in the same directory as this notebook:

# %%
# !ls *.py

# %%
# Configure IOP4 and logging (see examples)
# %run 01_notebook_configuration.py

# %% [markdown]
# For example, if you want to query the DB for the last night of observations, you can do:

# %%
from iop4lib.enums import *
from iop4lib.db import *

Epoch.objects.order_by('-night').first()

# %% [markdown]
For more information about querying the DB, see the Django ORM documentation.