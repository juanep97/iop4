# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# other imports
import os
import sys
import pytest
import yaml
import hashlib
from pathlib import Path

TEST_CONFIG = str(Path(iop4conf.datadir) / "config.tests.yaml")
TESTDATA_FPATH = str(Path("~/iop4testdata.tar.gz").expanduser())
TESTDATA_MD5SUM = '4d393377f8c659e2ead2fa252a9a38b2'
TEST_DATADIR = str(Path(iop4conf.datadir) / "iop4testdata")
TEST_DB_PATH = str(Path(iop4conf.db_path).expanduser().parent / ("test_" + str(Path(iop4conf.db_path).name)))

def pytest_configure():

    from django.conf import settings

    settings.configure(
        INSTALLED_APPS=[
            'iop4api',
        ],
        DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": iop4conf.db_path,
                "TEST": {"NAME":TEST_DB_PATH},
            }
        },
        DEBUG = False,
    )
    
    iop4conf.configure(db_path=TEST_DB_PATH, datadir=TEST_DATADIR)
    iop4conf.config_path = TEST_CONFIG
    iop4conf.datadir = TEST_DATADIR
    iop4conf.db_path = TEST_DB_PATH
    
@pytest.fixture(scope="session")
def testdata(request):
    setUpClass()
    request.addfinalizer(tearDownClass)
    
def setUpClass():

    # check if test data is available
    if not os.path.exists(TESTDATA_FPATH):
        raise Exception("Test dataset not found")
    
    # check if test data is the right version
    with open(Path('~/iop4testdata.tar.gz').expanduser(), 'rb') as f:
        if TESTDATA_MD5SUM != hashlib.md5(f.read()).hexdigest():
            raise Exception("Test dataset version is not right (md5 sum mismatch)")

    # prepare test data dir
    try:
        os.makedirs(Path(iop4conf.datadir).parent, exist_ok=True)
    except Exception as e:
        raise Exception(f"Error creating test data directory: {e}")
    
    # unpack test data
    if os.system(f"tar -xzf {TESTDATA_FPATH} -C {Path(iop4conf.datadir).parent}") != 0:
        raise Exception("Error unpacking test dataset")
    
    # create test config file
    try:
        with open(TEST_CONFIG, 'w') as f:
            f.write(yaml.safe_dump(dict(iop4conf)))
    except Exception as e:
        raise Exception(f"Error creating test config file: {e}")

    # check if test data dir is the right one, else exist inmediately (to avoid deleting wrong dir)
    if Path(iop4conf.datadir).name != "iop4testdata":
        print("DANGER!")
        sys.exit(-1)

def tearDownClass():
    # check if test data dir is the right one, else exist inmediately (to avoid deleting wrong dir)
    if Path(iop4conf.datadir).name != "iop4testdata":
        print("DANGER!")
        sys.exit(-1)
    # remove test data dir
    os.system(f"rm -rf {iop4conf.datadir}")
