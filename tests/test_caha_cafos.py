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
def test_build_multi_proc_photopol(load_test_catalog):
    """ Test the whole building process of reduced fits through multiprocessing 
    
    Also tests here relative photometry and polarimetry results and their 
    quality (value + uncertainties) (to avoid losing time reducing them
    in another test function).
    """

    from iop4lib.db import Epoch, RawFit, ReducedFit
    from iop4lib.enums import IMGTYPES, INSTRUMENTS

    # get epochs that have CAFOS observations in them
    from iop4lib.iop4 import list_local_epochnames
    epochname_L = list_local_epochnames()
    epoch_L = [Epoch.create(epochname=epochname) for epochname in epochname_L]
    epoch_L = [epoch for epoch in epoch_L if epoch.rawfits.filter(instrument=INSTRUMENTS.CAFOS).exists()]

    for epoch in epoch_L:
        epoch.build_master_biases()
        epoch.build_master_flats()

    # 1. Test multi-process reduced fits building

    iop4conf.nthreads = 4

    rawfits = RawFit.objects.filter(epoch__in=epoch_L, instrument=INSTRUMENTS.CAFOS, imgtype=IMGTYPES.LIGHT).all()
    
    Epoch.reduce_rawfits(rawfits)

    assert (ReducedFit.objects.filter(epoch__in=epoch_L).count() == len(rawfits)), "nº reduced != nº raw fits"

    for redf in ReducedFit.objects.filter(epoch__in=epoch_L).all():
        assert (redf.has_flag(ReducedFit.FLAGS.BUILT_REDUCED)), "redf is not BUILT_REDUCED"
        assert not (redf.has_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY)), "redf is ERROR_ASTROMETRY"

    from iop4lib.db import PhotoPolResult

    # 2. Test relative photo-polarimetry results

    # See `build_test_dataset.py`.

    # 2.1. "CAHA-T220/2025-09-13/caf-20250913-21:43:38-sci-agui.fits"

    # <PhotoPolResult(id: 834329
    #     reducedfits: [298086, 298087, 298088, 298089]
    #     CAFOS2.2 POLARIMETRY R / Hiltner960
    #     JD: 2460932.40608 (2025-09-13T21:44:45
    #     mag R: 9.785 ± 0.020
    #     p: (5.187 ± 0.248)%
    #     chi: (54.119 ± 1.379)º)>

    # Note: the magnitude check for this source has some caveats (see the 
    # comments in the test catalog file).

    epoch = Epoch.by_epochname("CAHA-T220/2025-09-13")
    epoch.compute_relative_polarimetry()

    r = PhotoPolResult.objects.filter(
        epoch=epoch,
        astrosource__name="Hiltner960",
    ).get()
    
    mag_R_lit = r.astrosource.mag_R
    p_lit = r.astrosource.p
    chi_lit = r.astrosource.chi

    assert r.mag == approx(mag_R_lit, abs=0.1), "mag_R within 0.1 of lit. value"
    assert r.mag == approx(mag_R_lit, abs=r.mag_err), "mag_R within mag_err of lit. value"
    assert r.mag_err < 0.1, "dmag < 0.1"

    assert r.p == approx(p_lit, abs=0.5/100), "p (%) within 0.5 of lit. value"
    assert r.p == approx(p_lit, abs=r.p_err), "p (%) within p_err of lit. value"
    assert r.p_err < 0.5/100, "dp < 0.5%"

    assert r.chi == approx(chi_lit, abs=3), "chi (º) within 3º of lit. value"
    assert r.chi == approx(chi_lit, abs=r.chi_err), "chi (º) within chi_err of lit.value"
    assert r.chi_err < 3, "dchi < 3º"

    # ensure also that reference values didn't change since included in tests
    assert mag_R_lit == approx(9.786)
    assert p_lit == approx(5.21/100)
    assert chi_lit == approx(54.54)
