""" Configuration for pytest tests.

Django pytest tests require giving the real database path. Also, we would like 
to use the configured astrometry index files path, if it was already configured 
in the system were tests are being run, to avoid downloading them again 
unnecessarily.

The configuration involves reading the real IOP4 config, changing the datadir 
and database paths to the test ones, and creating a test config file. It will 
also check whether the right version of the test dataset is available, otherwise
it will download it. Test submodules must be configured to use the created test
config file, e.g. with
```
import iop4lib.config
iop4conf = iop4lib.Config(config_path=TEST_CONFIG)
```

The test datadir is removed after the tests are run.
"""

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

TESTDATA_MD5SUM = '618810355ddd8997b31b9aa56f26f8fc'
TESTDATA_FPATH = str(Path(f"~/iop4testdata.{TESTDATA_MD5SUM}.tar.gz").expanduser())
TEST_CONFIG = str(Path("~/iop4testdata/config.tests.yaml").expanduser())
TEST_DATADIR = str(Path("~/iop4testdata").expanduser())
TEST_DB_PATH = str(Path("~/iop4testdata/test.db").expanduser())

def pytest_configure():

    from django.conf import settings

    settings.configure(
        INSTALLED_APPS=[
            'iop4api',
            # required to test the web interface, so the DB has user table
            "iop4admin",
            "django.contrib.admin",
            "django.contrib.auth",
            "django.contrib.contenttypes",
            "django.contrib.sessions",
            "django.contrib.messages",
            "django.contrib.staticfiles",
        ],
        DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": iop4conf.db_path,
                "TEST": {"NAME":TEST_DB_PATH},
            }
        },
        DEBUG = False,
        SECRET_KEY = "fake-test-key",
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

    # check if test data is available; otherwise download it
    if not os.path.exists(TESTDATA_FPATH):
        print("Downloading test dataset")
        import requests
        with requests.get(f"https://vhega.iaa.es/iop4/testdata/{Path(TESTDATA_FPATH).name}", stream=True) as r:
            r.raise_for_status()
            with open(TESTDATA_FPATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=None):
                    f.write(chunk)        

    # prepare test data dir
    try:
        os.makedirs(Path(iop4conf.datadir).parent, exist_ok=True)
    except Exception as e:
        raise Exception(f"Error creating test data directory: {e}")
    
    # unpack test data
    if os.system(f"tar -xzf {TESTDATA_FPATH} -C {Path(TEST_DATADIR).parent}") != 0:
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
