import pytest
from pathlib import Path

from .conftest import TEST_CONFIG

# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_path=TEST_CONFIG)

# other imports
import os
from pytest import approx

# logging
import logging
logger = logging.getLogger(__name__)

# fixtures
from .fixtures import load_test_catalog

@pytest.mark.django_db(transaction=True)
def test_epoch_creation(load_test_catalog):
    """ Test epoch creation """

    from iop4lib.db import Epoch

    assert (Epoch.objects.count() == 0)
    
    epoch = Epoch.create(epochname="OSN-T090/2022-09-08")

    assert (Epoch.objects.count() == 1)
    assert (epoch.rawfits.count() > 0)
    assert (epoch.rawfits.count() == len(os.listdir(Path(iop4conf.datadir) / "raw" / "OSN-T090" / "2022-09-08")))

@pytest.mark.django_db(transaction=True)
def test_epoch_masterbias_masterflats(load_test_catalog):
    """ Test masterbias and masterflats creation """
    from iop4lib.db import Epoch
    
    epoch = Epoch.create(epochname="OSN-T090/2022-09-23")

    assert (epoch.rawfits.count() == len(os.listdir(Path(iop4conf.datadir) / "raw" / "OSN-T090" / "2022-09-23")))
   
    epoch.build_master_biases()
    epoch.build_master_flats()

    assert (epoch.masterbias.count() > 0)
    assert (epoch.masterflats.count() > 0)


@pytest.mark.skip(reason="Not implemented yet")
@pytest.mark.django_db(transaction=True)
def test_polarimetry_groups(load_test_catalog):
    r""" Tests the splitting of polarimetry observations into groups.
    
    Organizing observations into groups is essential to derive polarimetry results.

    For OSN-T090 POLARIMETRY observations with AndorT090 instrument, four observations
    are needed to derive a single polarimetry result, for the same source, same band and same exptime,
    but different polarization angle.

    """

    assert False

@pytest.mark.django_db(transaction=True)
def test_build_single_proc(load_test_catalog):
    """ Test the whole building process of reduced fits in a single process """

    from iop4lib.db import Epoch, ReducedFit

    epochname_L = ["OSN-T090/2022-09-23", "OSN-T090/2022-09-18"]
    epoch_L = [Epoch.create(epochname=epochname, check_remote_list=False) for epochname in epochname_L]

    for epoch in epoch_L:
        epoch.build_master_biases()
        epoch.build_master_flats()

    iop4conf.nthreads = 1

    epoch = Epoch.by_epochname("OSN-T090/2022-09-18")
    
    epoch.reduce()

    assert (ReducedFit.objects.filter(epoch=epoch).count() == 1)

    for redf in ReducedFit.objects.filter(epoch=epoch).all():
        assert (redf.has_flag(ReducedFit.FLAGS.BUILT_REDUCED))
        assert not (redf.has_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY))


@pytest.mark.django_db(transaction=True)
def test_build_multi_proc_photopol(load_test_catalog):
    """ Test the whole building process of reduced fits through multiprocessing 
    
    Also tests here relative photometry and polarimetry results and their 
    quality (value + uncertainties) (to avoid losing time reducing them
    in another test function).

    See build_test_dataset.py for the test dataset used in this test.

    """

    from iop4lib.db import Epoch, RawFit, ReducedFit
    from iop4lib.enums import IMGTYPES, SRCTYPES, INSTRUMENTS

    # get epochs that have Andor90 observations in them
    from iop4lib.iop4 import list_local_epochnames
    epochname_L = list_local_epochnames()
    epoch_L = [Epoch.create(epochname=epochname) for epochname in epochname_L]
    epoch_L = [epoch for epoch in epoch_L if epoch.rawfits.filter(instrument=INSTRUMENTS.AndorT90).exists()]

    for epoch in epoch_L:
        epoch.build_master_biases()
        epoch.build_master_flats()

    # 1. Test multi-process reduced fits building

    iop4conf.nthreads = 6

    rawfits = RawFit.objects.filter(epoch__in=epoch_L, instrument=INSTRUMENTS.AndorT90, imgtype=IMGTYPES.LIGHT).all()
    
    Epoch.reduce_rawfits(rawfits)

    assert (ReducedFit.objects.filter(epoch__in=epoch_L).count() == 5)

    for redf in ReducedFit.objects.filter(epoch__in=epoch_L).all():
        assert (redf.has_flag(ReducedFit.FLAGS.BUILT_REDUCED))
        assert not (redf.has_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY))

    from iop4lib.db import PhotoPolResult, AstroSource

    # 2. Test relative photometry 

    logger.info("Testing relative photometry results")

    epoch = Epoch.by_epochname("OSN-T090/2022-09-18")

    epoch.compute_relative_photometry()

    qs_res = PhotoPolResult.objects.filter(epoch=epoch).all()

    # we expect only one photometry result target in this test dataset for this epoch
    assert qs_res.exclude(astrosource__in=AstroSource.objects.filter(is_calibrator=True)).count() == 1

    res = qs_res.get(astrosource__name="2200+420")

    # check that the photometric result is correct to 1.5 sigma or 0.02 mag compared to IOP3
    assert res.mag == approx(13.35, abs=max(1.5*res.mag_err, 0.05))

    # check that uncertainty of the photometric result is less than 0.06 mag
    assert res.mag_err < 0.06

    # 3. Test relative polarimetry

    logger.info("Testing relative polarimetry results")

    epoch = epoch.by_epochname("OSN-T090/2022-09-08")

    epoch.compute_relative_polarimetry()

    qs_res = PhotoPolResult.objects.filter(epoch=epoch).all()

    # we expect only one polarimetry result target in this test dataset for this epoch
    assert qs_res.exclude(astrosource__in=AstroSource.objects.filter(is_calibrator=True)).count() == 1

    res = qs_res.get(astrosource__name="2200+420")

    # check that the magnitude and polarization of the result relative to IOP3
    # for polarimetry, we expect a higher uncertainty than for photometry

    assert res.mag == approx(13.38, abs=max(1.5*res.mag_err, 0.05)) # 1.5 sigma or 0.05 mag
    assert res.mag_err < 0.1

    assert res.p == approx(14.0/100, abs=max(2*res.p_err, 2/100)) # 2 sigma or 2% of polarization degree
    assert res.p_err < 0.5/100

    assert res.chi == approx(14.7, abs=max(2*res.chi_err, 5)) # 2 sigma or 5 degrees of polarization angle
    assert res.chi_err < 2.5
