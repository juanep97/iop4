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


@pytest.mark.skipif(os.getenv("CI") != "true", reason="only for actions CI")
@pytest.mark.django_db(transaction=True)
def test_build_single_proc(load_test_catalog):
    """ Test the whole building process of reduced fits in a single process

        This test is not really necessary since single-process reduction is already tested
        in OSN-T090, therefore it is skipped by default. However CI actions at the momemnt 
        will fail without it, because if the astrometry index files are accessed by multiple 
        processes at the same time before they are catched httpdirfs will fail. To run it 
        locally set the environment variable CI=true.
    """

    from iop4lib.db import Epoch, RawFit, ReducedFit
    from iop4lib.enums import IMGTYPES, SRCTYPES

    epochname_L = ["CAHA-T220/2022-08-27", "CAHA-T220/2022-09-18"]

    epoch_L = [Epoch.create(epochname=epochname, check_remote_list=False) for epochname in epochname_L]

    for epoch in epoch_L:
        epoch.build_master_biases()
        epoch.build_master_flats()

    iop4conf.nthreads = 1

    rawfits = RawFit.objects.filter(epoch__in=epoch_L, imgtype=IMGTYPES.LIGHT).all()
    
    Epoch.reduce_rawfits(rawfits)

    assert (ReducedFit.objects.filter(epoch__in=epoch_L).count() == 4)

    for redf in ReducedFit.objects.filter(epoch__in=epoch_L).all():
        assert (redf.has_flag(ReducedFit.FLAGS.BUILT_REDUCED))
        assert not (redf.has_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY))



@pytest.mark.django_db(transaction=True)
def test_build_multi_proc_photopol(load_test_catalog):
    """ Test the whole building process of reduced fits through multiprocessing 
    
    Also tests here relative photometry and polarimetry results and their 
    quality (value + uncertainties) (to avoud losing time reducing them
    in another test function).
    """

    from iop4lib.db import Epoch, RawFit, ReducedFit
    from iop4lib.enums import IMGTYPES, SRCTYPES

    epochname_L = ["CAHA-T220/2022-08-27", "CAHA-T220/2022-09-18"]

    epoch_L = [Epoch.create(epochname=epochname, check_remote_list=False) for epochname in epochname_L]

    for epoch in epoch_L:
        epoch.build_master_biases()
        epoch.build_master_flats()

    iop4conf.nthreads = 4

    rawfits = RawFit.objects.filter(epoch__in=epoch_L, imgtype=IMGTYPES.LIGHT).all()
    
    Epoch.reduce_rawfits(rawfits)

    assert (ReducedFit.objects.filter(epoch__in=epoch_L).count() == 4)

    for redf in ReducedFit.objects.filter(epoch__in=epoch_L).all():
        assert (redf.has_flag(ReducedFit.FLAGS.BUILT_REDUCED))
        assert not (redf.has_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY))

    from iop4lib.db import PhotoPolResult, AstroSource

    epoch = Epoch.by_epochname("CAHA-T220/2022-09-18")

    epoch.compute_relative_polarimetry()

    qs_res = PhotoPolResult.objects.filter(epoch=epoch, astrosource__name="2200+420").all()

    # we expect only one photometry result target in this test dataset for this epoch and source
    assert qs_res.exclude(astrosource__srctype=SRCTYPES.CALIBRATOR).count() == 1

    res = qs_res[0]

    # check that the result is correct to 1.5 sigma or 0.02 mag compared to IOP3
    assert res.mag == approx(13.38, abs=max(1.5*res.mag_err, 0.05))
    # check that uncertainty of the result is less than 0.08 mag
    assert res.mag_err < 0.08

    assert res.p == approx(10.9/100, abs=max(1.5*res.p_err, 1.0/100))
    assert res.p_err < 0.5

    assert res.chi == approx(25.2, abs=max(1.5*res.chi_err, 1.0))
    assert res.chi_err < 1.0
