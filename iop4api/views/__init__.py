from django.shortcuts import render

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.contrib.auth.decorators import login_required
from django.template import loader

from django.contrib.admin.views.decorators import staff_member_required

from ..models import *
# from .. import admin


# IOP4API 

from .others import index, login_view, logout_view, catalog, data
from .plot import plot

# IOP4ADMIN VIEWs

