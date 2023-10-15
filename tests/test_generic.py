import pytest
from pathlib import Path

from .conftest import TEST_CONFIG

# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_path=TEST_CONFIG)

# other imports
import os

# logging
import logging
logger = logging.getLogger(__name__)

# fixtures
from .fixtures import load_test_catalog


@pytest.mark.django_db(transaction=True)
def test_testdata(testdata):
    """Test that the test data is available"""
    assert (os.path.exists(Path(iop4conf.datadir) / "raw" / "OSN-T090"))


@pytest.mark.django_db(transaction=True)
def test_testconfig_testdb(load_test_catalog):
    """ Check that the DB is clean (it should be the test database), if it is not, all test will fail """
    from iop4lib.db import Epoch, RawFit, ReducedFit, MasterBias, MasterFlat, AperPhotResult, PhotoPolResult, AstroSource
    assert (hasattr(iop4conf, "basedir"))
    assert (hasattr(iop4conf, "datadir"))
    assert (hasattr(iop4conf, "db_path"))
    assert (hasattr(iop4conf, "config_path"))
    assert (Path(iop4conf.datadir).name == "iop4testdata")
    assert (Path(iop4conf.config_path).name == "config.tests.yaml")
    assert ("test_" in Path(iop4conf.db_path).name)
    assert (Epoch.objects.count() == 0)
    assert (RawFit.objects.count() == 0)
    assert (ReducedFit.objects.count() == 0)
    assert (MasterBias.objects.count() == 0)
    assert (MasterFlat.objects.count() == 0)
    assert (AperPhotResult.objects.count() == 0)
    assert (PhotoPolResult.objects.count() == 0)

    # there should be some test sources in the DB, and their calibrators
    assert (0 < AstroSource.objects.count() < 20)
    assert AstroSource.objects.filter(name="2200+420").exists()