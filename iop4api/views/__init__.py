from django.shortcuts import render

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.contrib.auth.decorators import login_required
from django.template import loader

from django.contrib.admin.views.decorators import staff_member_required

from ..models import *
# from .. import admin


# IOP4API 

from .index import index
from .plot import plot
from .catalog import catalog
from .data import data
from .auth import login_view, logout_view
from .log import log
from .flag import flag
from .docs import docs
