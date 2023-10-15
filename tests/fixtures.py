import pytest
from pathlib import Path

from .conftest import TEST_CONFIG

# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_path=TEST_CONFIG)

# django imports

# other imports
import os
from pytest import approx

# logging
import logging
logger = logging.getLogger(__name__)

@pytest.fixture
def load_test_catalog(testdata, django_db_setup, django_db_blocker):
    with django_db_blocker.unblock():
        from iop4lib.db import AstroSource

        # load test catalog in test db
        from django.core.management import call_command
        call_command('loaddata', str(Path(iop4conf.datadir) / 'testcatalog.yaml'), verbosity=0)
