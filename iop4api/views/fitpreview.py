from django.shortcuts import render

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound, FileResponse
from django.template import loader

from django.apps import apps

from ..models import *
from .. import admin

import base64

import logging
logger = logging.getLogger(__name__)

def view_fitpreview(site, request, fitclsname, id):  
    """ API endpoint to download a png preview from FITS file.

    Parameters
    ----------
        site : (automatically passed) admin site
        request : (automatically passed) request
        fitclsname : str 
            name of the fit class (RawFit, MasterFlat, MasterBias, ReducedFit)
        id : int
            id or primary key of the file.
        
    """
    
    try:
        modelcls = apps.get_model('iop4api', fitclsname)
    except LookupError:
        return HttpResponseNotFound()
    
    if ((fit := modelcls.objects.filter(id=id).first()) is None):
        return HttpResponseNotFound()

    if not fit.fileexists:
        return HttpResponseNotFound()

    imgbytes = fit.get_imgbytes_preview_image()
    return HttpResponse(imgbytes, content_type="image/png")