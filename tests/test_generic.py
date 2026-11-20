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
    assert ("test" in Path(iop4conf.db_path).name)
    assert (Epoch.objects.count() == 0)
    assert (RawFit.objects.count() == 0)
    assert (ReducedFit.objects.count() == 0)
    assert (MasterBias.objects.count() == 0)
    assert (MasterFlat.objects.count() == 0)
    assert (AperPhotResult.objects.count() == 0)
    assert (PhotoPolResult.objects.count() == 0)

    # there should be some test sources in the DB, and their calibrators
    assert AstroSource.objects.count() > 0

@pytest.mark.django_db(transaction=True)
def test_catalog(load_test_catalog):
    """Test that test catalog
    Check that reference values we are using in other tests did not change since
    we included them.
    """
    from iop4lib.db import AstroSource
    src = AstroSource.objects.get(name="Hiltner960")
    assert src.mag_R == approx(9.786)
    assert src.p == approx(5.21/100)
    assert src.chi == approx(54.54)

def test_astrosource_metadata_validation():
    """ Check the validation of AstroSource metadata keys and values """
    from iop4lib.db import AstroSource

    for valid in [
        {"DIPOL.polarimetry.only_pair": "O"},
        {"DIPOL.polarimetry.sigma_clip": False},
        {"mask_other_sources": True},
        {"other_masks": [{"ra": "10h31m22.21s", "dec": "+74d41m54.28s", "r": "3 arcsec"}]},
        {"fit_centroid": False},
        {"aperture_radius": "2 arcsec"},
    ]:
        AstroSource.validate_metadata(valid)

    for invalid in [
        {"DIPOL.polarimetry.only_pair": "X"},
        {"DIPOL.polarimetry.sigma_clip": "no"},
        {"mask_other_sources": "no"},
        {"fit_centroid": "no"},
        {"other_masks": [{"ra": "10h31m22.21s", "dec": "+74d41m54.28s"}]},
        {"aperture_radius": "2 pix"},
        {"unknown_key": True},
    ]:
        with pytest.raises(ValueError):
            AstroSource.validate_metadata(invalid)

    with pytest.raises(TypeError):
        AstroSource.validate_metadata(["fit_centroid"])

def test_polarimetry_sigma_clip():
    """ The polarimetry fit discards outliers, unless called with sigma_clip=False """
    import numpy as np
    from iop4lib.utils.polarization import compute_stokes_HWP_fit_rel_nonideal

    theta = np.arange(0, 360, 22.5)
    F = 0.03*np.cos(np.deg2rad(4*theta)) - 0.02*np.sin(np.deg2rad(4*theta))
    F = F + np.random.default_rng(0).normal(0, 0.002, len(theta))
    FO, FE = 1e4*(1+F), 1e4*(1-F)
    FO[3] *= 1.2 # outlier
    dFO = dFE = np.full(len(theta), 20.0)

    _, stats_clip = compute_stokes_HWP_fit_rel_nonideal(theta=theta, FO=FO, FE=FE, dFO=dFO, dFE=dFE)
    _, stats_noclip = compute_stokes_HWP_fit_rel_nonideal(theta=theta, FO=FO, FE=FE, dFO=dFO, dFE=dFE, sigma_clip=False)

    q_clip, u_clip, _ = stats_clip["popt"]
    q_noclip, u_noclip, _ = stats_noclip["popt"]

    # the outlier biases the fit only if it is not discarded
    assert np.hypot(q_clip-0.03, u_clip+0.02) < 0.003
    assert np.hypot(q_noclip-0.03, u_noclip+0.02) > 0.005

@pytest.mark.django_db(transaction=True)
def test_host_correction_data_load(load_test_catalog):
    r""" Check that the host correction data can be correctly loaded """
    from iop4lib.db import AstroSource
    from iop4lib.utils.photometry import get_host_correction

    val, err = get_host_correction(AstroSource.objects.get(name="2200+420"), 6, 4)

    assert val is not None
    assert val == approx(1.22)
    assert err == approx(0.03)

@pytest.mark.django_db(transaction=True)
def test_iop4_script():
    """ Test that the iop4 script is available """
    import subprocess
    assert subprocess.run(["iop4", "--help"]).returncode == 0, "iop4 help invokation"
    assert subprocess.run(["iop4", "--check-config"]).returncode == 0, "iop4 configuration check"


def _load_version_module_from(dest):
    """ Copy iop4lib/version.py to `dest` and import it from there """
    import shutil
    import importlib.util
    import iop4lib.version

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(iop4lib.version.__file__, dest)

    spec = importlib.util.spec_from_file_location("version_copy", dest)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_version_without_git(tmp_path, monkeypatch):
    """ Without git (e.g. non-editable installs), use the installed version.

    Regression test for PR #141 (git was required at import time).
    """
    monkeypatch.setenv("PATH", "")
    version = _load_version_module_from(tmp_path / "iop4lib" / "version.py")

    assert version.GIT_COMMIT_HASH is None
    assert version.GIT_BRANCH is None
    assert version.GIT_DESCRIBE == version.__version__

def test_version_in_unrelated_git_repo(tmp_path):
    """ The git info of an unrelated repo containing IOP4 must not be used.

    See `test_version_without_git`.
    """
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test",
         "commit", "-q", "--allow-empty", "-m", "init"],
        cwd=tmp_path, check=True,
    )

    dest = tmp_path / "venv" / "site-packages" / "iop4lib" / "version.py"
    version = _load_version_module_from(dest)

    assert version.GIT_COMMIT_HASH is None
    assert version.GIT_DESCRIBE == version.__version__
