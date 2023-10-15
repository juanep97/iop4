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
def test_build_multi_proc(load_test_catalog):
    """ Test the whole building process of reduced fits through multiprocessing """

    from iop4lib.db import Epoch, RawFit, ReducedFit
    from iop4lib.enums import IMGTYPES, SRCTYPES

    epochname_L = ["CAHA-T220/2022-09-18", "CAHA-T220/2022-08-27"]

    epoch_L = [Epoch.create(epochname=epochname, check_remote_list=False) for epochname in epochname_L]

    for epoch in epoch_L:
        epoch.build_master_biases()
        epoch.build_master_flats()

    # workaround for CI
    # otherwise the attempt to access the httpdsdir-mounted files directly through multiprocessing will fail
    if os.getenv("CI") == 'true':
        iop4conf.max_concurrent_threads = 1
        Epoch.reduce_rawfits([RawFit.objects.filter(epoch__in=epoch_L, imgtype=IMGTYPES.LIGHT).first()])

    iop4conf.max_concurrent_threads = 4

    rawfits = RawFit.objects.filter(epoch__in=epoch_L, imgtype=IMGTYPES.LIGHT).all()
    
    Epoch.reduce_rawfits(rawfits)

    assert (ReducedFit.objects.filter(epoch__in=epoch_L).count() == 4)

    for redf in ReducedFit.objects.filter(epoch__in=epoch_L).all():
        assert (redf.has_flag(ReducedFit.FLAGS.BUILT_REDUCED))
        assert not (redf.has_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY))

    from iop4lib.db import PhotoPolResult, AstroSource

    epoch = Epoch.by_epochname("CAHA-T220/2022-09-18")

    epoch.compute_relative_photometry()
    epoch.compute_relative_polarimetry()

    qs_res = PhotoPolResult.objects.filter(epoch=epoch, astrosource__name="2200+420").all()

    # we expect only one photometry result target in this test dataset for this epoch and source
    assert qs_res.exclude(astrosource__srctype=SRCTYPES.CALIBRATOR).count() == 1

    res = qs_res[0]

    # check that the result is correct to 1.5 sigma compared to IOP3
    assert res.mag == approx(13.38, abs=1.5*res.mag_err)

    # check that uncertainty of the result is less than 0.08 mag
    assert res.mag_err < 0.08