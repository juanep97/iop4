import pytest
from .conftest import TEST_CONFIG

# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_path=TEST_CONFIG)

# other imports
import re
from pathlib import Path
from django.test import client
from django.test import override_settings

# logging
import logging
logger = logging.getLogger(__name__)

# we need to make iop4site module as iop4site, not as iop4site.iop4site
# we also need to load all the rest of settings that we usually don't use
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "iop4site/"))
import iop4site.settings
settings_dict = {key:value for key,value in vars(iop4site.settings).items() if re.match(r'^[A-Z][A-Z_0-9]*$', key)}
settings_dict["ALLOWED_HOSTS"] = ['testserver']

@override_settings(**settings_dict)
def test_index(client):
    """Test the index page"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Juan Escudero Pedrosa" in response.content