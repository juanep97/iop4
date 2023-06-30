# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.shortcuts import render
from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.template import loader
from django.apps import apps
from django.views.generic.base import RedirectView

# iop4api imports
from ..models import *
from .. import admin

#logging
import logging
logger = logging.getLogger(__name__)

def index(request):
    return HttpResponseRedirect(reverse('iop4admin:index'))