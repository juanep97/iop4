from django.shortcuts import render

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.contrib.auth.decorators import login_required
from django.template import loader

from django.contrib.admin.views.decorators import staff_member_required

from ..models import *
from .. import admin


# IOP4API 

from .index import *

# IOP4ADMIN VIEWs

from .fitpreview import *
from .getfile import *
from .fitviewer import *
from .fitdetails import *
from .epochdetails import *
from .astrosourcedetails import *