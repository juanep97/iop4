from django.shortcuts import render

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound, FileResponse
from django.template import loader

from django.apps import apps

from ..models import *
from .. import admin

import logging
logger = logging.getLogger(__name__)

def view_getfile(site, request, filecls, id):  
    """ API endpoint to download a file registered as a model in the database.

    Parameters
    ----------
        site : (automatically passed) admin site
        request : (automatically passed) request
        filecls : str
            name of the file class (RawFit, etc).
        id : int
            id or primary key of the file.

    Notes
    -----
    Any model that has a .fileexists property and a .filepath property can be requested, the file
    at path .filepath will be returned.

    """
    
    try:
        modelcls = apps.get_model('iop4api', filecls)
    except LookupError:
        return HttpResponseNotFound()
    
    if ((fit := modelcls.objects.filter(id=id).first()) is None):
        return HttpResponseNotFound()

    if not fit.fileexists:
        return HttpResponseNotFound()

    return FileResponse(open(fit.filepath, 'rb'), filename=fit.filename)