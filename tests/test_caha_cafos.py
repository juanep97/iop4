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

    # <PhotoPolResult(id: 883611
    #     reducedfits: [298086, 298087, 298088, 298089]
    #     CAFOS2.2 POLARIMETRY R / Hiltner960
    #     JD: 2460932.40608 (2025-09-13T21:44:45)
    #     mag R: 9.797 ± 0.012
    #     p: (5.116 ± 0.271)%
    #     chi: (53.772 ± 2.127)º)>

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
    assert r.mag == approx(mag_R_lit, abs=1.25*r.mag_err), "mag_R within ~mag_err of lit. value"
    assert r.mag_err < 0.1, "dmag < 0.1"

    assert r.p == approx(p_lit, abs=0.5/100), "p (%) within 0.5 of lit. value"
    assert r.p == approx(p_lit, abs=1.25*r.p_err), "p (%) within ~p_err of lit. value"
    assert r.p_err < 0.5/100, "dp < 0.5%"

    assert r.chi == approx(chi_lit, abs=3), "chi (º) within 3º of lit. value"
    assert r.chi == approx(chi_lit, abs=1.25*r.chi_err), "chi (º) within ~chi_err of lit.value"
    assert r.chi_err < 3, "dchi < 3º"

    # ensure also that reference values didn't change since included in tests
    assert mag_R_lit == approx(9.786)
    assert p_lit == approx(5.21/100)
    assert chi_lit == approx(54.54)


class FakeSFTP:
    """ Fake SFTP client for a CAHA account, with remote dirs {dirname: [fnames]}. """

    def __init__(self, user, dirs):
        self.user = user
        self.dirs = dirs
        self.cwd = None

    def chdir(self, dirname):
        self.cwd = dirname

    def listdir(self, path='.'):
        dirname = self.cwd if path == '.' else path
        return list(self.dirs) if dirname is None else list(self.dirs[dirname])

    def get(self, remotepath, localpath):
        Path(localpath).write_text(f"{self.user}:{remotepath}")

    def close(self):
        pass


@pytest.fixture
def fake_caha_archive(monkeypatch, tmp_path):
    """ Fake CAHA archive, returns a function to set the remote dirs of each account. """

    from iop4lib.telescopes import CAHAT220

    monkeypatch.setattr(iop4conf, "datadir", str(tmp_path))
    monkeypatch.setattr(iop4conf, "caha_address", "fake.caha.es:22")

    remote = dict()

    def set_remote(accounts):
        """ accounts: {(user, password): {dirname: [fnames]}} """
        remote.clear()
        remote.update(accounts)
        monkeypatch.setattr(iop4conf, "caha_accounts", [{"user": user, "password": password} for user, password in accounts], raising=False)

    monkeypatch.setattr(CAHAT220, "_sftp_connect", classmethod(lambda cls, user, password: FakeSFTP(user, remote[(user, password)])))

    return set_remote


def test_caha_multiple_accounts(fake_caha_archive):
    """ Test that epochs and files of all CAHA accounts are listed and downloaded from their account. """

    from types import SimpleNamespace
    from iop4lib.telescopes import CAHAT220

    fake_caha_archive({
        ("user1", "pass1"): {"250913_CAFOS": ["a.fits", "a.txt"]},
        ("user2", "pass2"): {"250914_CAFOS": ["b.fits"], "notanepoch": []},
    })

    assert CAHAT220.ftp_accounts == [("user1", "pass1"), ("user2", "pass2")]

    assert sorted(CAHAT220.list_remote_epochnames()) == ["CAHA-T220/2025-09-13", "CAHA-T220/2025-09-14"]

    assert CAHAT220.list_remote_raw_fnames(SimpleNamespace(epochname="CAHA-T220/2025-09-14")) == ["b.fits"]

    filelocs = CAHAT220.list_remote_filelocs(["CAHA-T220/2025-09-13", "CAHA-T220/2025-09-14", "CAHA-T220/2025-09-15"])
    assert sorted(filelocs) == ["CAHA-T220/2025-09-13/a.fits", "CAHA-T220/2025-09-14/b.fits"]

    CAHAT220.download_rawfits(filelocs)
    assert (Path(iop4conf.datadir) / "raw/CAHA-T220/2025-09-13/a.fits").read_text() == "user1:250913_CAFOS/a.fits"
    assert (Path(iop4conf.datadir) / "raw/CAHA-T220/2025-09-14/b.fits").read_text() == "user2:250914_CAFOS/b.fits"

    with pytest.raises(Exception, match="not found in any"):
        CAHAT220.download_rawfits(["CAHA-T220/2025-09-15/c.fits"])


def test_caha_single_account(fake_caha_archive):
    """ Test that a single CAHA account works (it doesn't need to log in to find the account of each dir). """

    from iop4lib.telescopes import CAHAT220

    fake_caha_archive({("user1", "pass1"): {"250913_CAFOS": ["a.fits"]}})

    assert CAHAT220.ftp_accounts == [("user1", "pass1")]
    assert CAHAT220.list_remote_epochnames() == ["CAHA-T220/2025-09-13"]

    CAHAT220.download_rawfits(["CAHA-T220/2025-09-13/a.fits"])
    assert (Path(iop4conf.datadir) / "raw/CAHA-T220/2025-09-13/a.fits").read_text() == "user1:250913_CAFOS/a.fits"


def test_caha_same_night_in_several_accounts(fake_caha_archive):
    """ Test that a night found in more than one CAHA account raises an error. """

    from iop4lib.telescopes import CAHAT220

    fake_caha_archive({
        ("user1", "pass1"): {"250913_CAFOS": ["a.fits"]},
        ("user2", "pass2"): {"250913_CAFOS": ["b.fits"]},
    })

    with pytest.raises(Exception, match="more than one"):
        CAHAT220.list_remote_epochnames()

    with pytest.raises(Exception, match="more than one"):
        CAHAT220.list_remote_filelocs(["CAHA-T220/2025-09-13"])

