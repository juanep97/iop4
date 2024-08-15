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
from django.contrib import auth

# logging
import logging
logger = logging.getLogger(__name__)

# we need to make iop4site module available as iop4site, not as iop4site.iop4site
# we also need to load all the rest of settings that we usually don't use
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "iop4site/"))
import iop4site.settings
settings_dict = {key:value for key,value in vars(iop4site.settings).items() if re.match(r'^[A-Z][A-Z_0-9]*$', key)}
# We want to test the production scenario
settings_dict["DEBUG"] = False
settings_dict["ALLOWED_HOSTS"] = ['testserver']

@override_settings(**settings_dict)
def test_index(client):
    """Test the index page"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Juan Escudero Pedrosa" in response.content

@override_settings(**settings_dict)
@pytest.mark.django_db(transaction=True)
def test_login(client):
    """Test the login and login page"""

    response = client.get('/iop4/login/')
    assert response.status_code == 200
    assert b"Username" in response.content
    assert b"Password" in response.content

    # create a test user
    from django.contrib.auth.models import User
    user = User.objects.create_user(username='testuser', password='testpassword')
    user.save()

    # try to login
    response = client.post('/iop4/api/login/', 
                           {'username': 'testuser', 'password': 'testpassword'},
                            follow=True)
    
    # check that the user is authenticated
    user = auth.get_user(client)
    assert user.is_authenticated

    # check that the explore tab is present
    client.get('/iop4/explore/')
    assert response.status_code == 200
    assert b"Explore data" in response.content

@override_settings(**settings_dict)
@pytest.mark.django_db(transaction=True)
def test_failed_login(client):
    """Test that the login fails with wrong credentials"""

    response = client.get('/iop4/login/')
    assert response.status_code == 200
    assert b"Username" in response.content
    assert b"Password" in response.content

    # create a test user
    from django.contrib.auth.models import User
    user = User.objects.create_user(username='testuser', password='testpassword')
    user.save()

    # try to login
    response = client.post('/iop4/api/login/', 
                           {'username': 'testuser', 'password': 'wrongtestpassword'},
                            follow=True)
    
    # check that the user is not authenticated
    user = auth.get_user(client)
    assert not user.is_authenticated

    # check that the explore tab is not present
    client.get('/iop4/explore/')
    assert response.status_code == 200
    assert not b"Explore data" in response.content