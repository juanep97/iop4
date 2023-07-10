import pytest
from pathlib import Path

TEST_CONFIG = str(Path('~/.iop4tests/').expanduser() / "config.tests.yaml")

# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_path=TEST_CONFIG)

# django imports

# other imports
import os

# logging
import logging
logger = logging.getLogger(__name__)

@pytest.mark.django_db(transaction=True)
def test_testconfig_testdb():
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
    assert (AstroSource.objects.count() == 0)

@pytest.mark.django_db(transaction=True)
def test_testdata(testdata):
    """Test that the test data is available"""
    assert (os.path.exists(Path(iop4conf.datadir) / "raw" / "OSN-T090"))

@pytest.mark.django_db(transaction=True)
def test_epoch_creation(testdata):
    """ Test epoch creation """

    from iop4lib.db import Epoch

    assert (Epoch.objects.count() == 0)
    
    epoch = Epoch.create(epochname="OSN-T090/2023-06-11")

    assert (Epoch.objects.count() == 1)
    assert (epoch.rawfits.count() > 0)
    assert (epoch.rawfits.count() == len(os.listdir(Path(iop4conf.datadir) / "raw" / "OSN-T090" / "2023-06-11")))

@pytest.mark.django_db(transaction=True)
def test_epoch_masterbias_masterflats(testdata):
    """ Test masterbias and masterflats creation """
    from iop4lib.db import Epoch
    
    epoch = Epoch.create(epochname="OSN-T090/2023-06-11")

    assert (epoch.rawfits.count() == len(os.listdir(Path(iop4conf.datadir) / "raw" / "OSN-T090" / "2023-06-11")))
   
    epoch.build_master_biases()
    epoch.build_master_flats()

    assert (epoch.masterbias.count() == 1)
    assert (epoch.masterflats.count() == 5)


@pytest.mark.django_db(transaction=True)
def test_build_reduced_multiproc(testdata):
    """ Test the whole building process of reduced fits through multiprocessing """
    from iop4lib.db import Epoch, ReducedFit

    epoch = Epoch.create(epochname="OSN-T090/2023-06-11", check_remote_list=False)
    epoch.build_master_biases()
    epoch.build_master_flats()

    iop4conf.max_concurrent_threads = 2

    mb = epoch.masterbias.first()
    mf = epoch.masterflats.first()
    rf = epoch.rawfits.first()

    epoch.reduce()

    assert (ReducedFit.objects.filter(epoch=epoch).count() == 5)

    for redf in ReducedFit.objects.filter(epoch=epoch).all():
        assert (redf.has_flag(ReducedFit.FLAGS.BUILT_REDUCED))
        assert not (redf.has_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY))


