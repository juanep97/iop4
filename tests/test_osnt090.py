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


@pytest.mark.django_db(transaction=True)
def test_testdata(testdata):
    """Test that the test data is available"""
    assert (os.path.exists(Path(iop4conf.datadir) / "raw" / "OSN-T090"))

@pytest.fixture
def load_test_catalog(testdata, django_db_setup, django_db_blocker):
    with django_db_blocker.unblock():
        from iop4lib.db import AstroSource

        # load test catalog in test db
        from django.core.management import call_command
        call_command('loaddata', str(Path(iop4conf.datadir) / 'testcatalog.yaml'), verbosity=0)

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

@pytest.mark.django_db(transaction=True)
def test_epoch_creation(load_test_catalog):
    """ Test epoch creation """

    from iop4lib.db import Epoch

    assert (Epoch.objects.count() == 0)
    
    epoch = Epoch.create(epochname="OSN-T090/2023-06-11")

    assert (Epoch.objects.count() == 1)
    assert (epoch.rawfits.count() > 0)
    assert (epoch.rawfits.count() == len(os.listdir(Path(iop4conf.datadir) / "raw" / "OSN-T090" / "2023-06-11")))

@pytest.mark.django_db(transaction=True)
def test_epoch_masterbias_masterflats(load_test_catalog):
    """ Test masterbias and masterflats creation """
    from iop4lib.db import Epoch
    
    epoch = Epoch.create(epochname="OSN-T090/2023-06-11")

    assert (epoch.rawfits.count() == len(os.listdir(Path(iop4conf.datadir) / "raw" / "OSN-T090" / "2023-06-11")))
   
    epoch.build_master_biases()
    epoch.build_master_flats()

    assert (epoch.masterbias.count() == 1)
    assert (epoch.masterflats.count() == 5)


@pytest.mark.django_db(transaction=True)
def test_build_single_proc(load_test_catalog):
    """ Test the whole building process of reduced fits in a single process """

    from iop4lib.db import Epoch, ReducedFit

    epoch = Epoch.create(epochname="OSN-T090/2023-06-11", check_remote_list=False)
    epoch.build_master_biases()
    epoch.build_master_flats()

    iop4conf.max_concurrent_threads = 1

    epoch.reduce()

    assert (ReducedFit.objects.filter(epoch=epoch).count() == 5)

    for redf in ReducedFit.objects.filter(epoch=epoch).all():
        assert (redf.has_flag(ReducedFit.FLAGS.BUILT_REDUCED))
        assert not (redf.has_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY))



@pytest.mark.django_db(transaction=True)
def test_build_multi_proc(load_test_catalog):
    """ Test the whole building process of reduced fits through multiprocessing """
    from iop4lib.db import Epoch, RawFit, ReducedFit
    from iop4lib.enums import IMGTYPES, SRCTYPES

    epochname_L = ["OSN-T090/2022-09-18", "OSN-T090/2023-06-11"]

    epoch_L = [Epoch.create(epochname=epochname, check_remote_list=False) for epochname in epochname_L]

    for epoch in epoch_L:
        epoch.build_master_biases()
        epoch.build_master_flats()

    iop4conf.max_concurrent_threads = 6

    rawfits = RawFit.objects.filter(epoch__in=epoch_L, imgtype=IMGTYPES.LIGHT).all()
    
    Epoch.reduce_rawfits(rawfits)

    assert (ReducedFit.objects.filter(epoch__in=epoch_L).count() == 6)

    for redf in ReducedFit.objects.filter(epoch__in=epoch_L).all():
        assert (redf.has_flag(ReducedFit.FLAGS.BUILT_REDUCED))
        assert not (redf.has_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY))

    from iop4lib.db import PhotoPolResult, AstroSource

    epoch = Epoch.by_epochname("OSN-T090/2022-09-18")

    epoch.compute_relative_photometry()

    qs_res = PhotoPolResult.objects.filter(epoch=epoch).all()

    # we expect only one photometry result target in this test dataset for this epoch
    assert qs_res.exclude(astrosource__srctype=SRCTYPES.CALIBRATOR).count() == 1

    res = qs_res[0]

    # check that the result is correct to 1.5 sigma compared to IOP3
    assert res.mag == approx(13.35, abs=1.5*res.mag_err)

    # check that uncertainty of the result is less than 0.08 mag
    assert res.mag_err < 0.08