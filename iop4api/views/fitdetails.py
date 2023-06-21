from django.shortcuts import render

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.contrib.auth.decorators import login_required
from django.template import loader

from django.contrib.admin.views.decorators import staff_member_required

from django.apps import apps

from ..models import *
from .. import admin

# other imports

import os
import numpy as np
import astropy.io.fits as fits
import base64
import glob

# logging

import logging
logger = logging.getLogger(__name__)

#@staff_member_required
# no staff_member_required, because it is only routed from iop4admin_site
def view_fitdetails(site, request, fitclsname, id):  

    # Add admin context to the template context

    context = dict(site.each_context(request))

    # If fitclsname is not a valid class, return early
    if fitclsname not in ["RawFit", "MasterFlat", "MasterBias", "ReducedFit"]:
        return HttpResponseNotFound()
    
    modelcls = apps.get_model('iop4api', fitclsname)

    # If id is not a valid id, return early

    if ((fit := modelcls.objects.filter(id=id).first()) is None):
        return HttpResponseRedirect(reverse('iop4admin:%s_%s_change' % (modelcls._meta.app_label, modelcls._meta.model_name), args=(id,)))

    # Add fit info to the template context

    ## page title
    context['title'] = f"{fitclsname} {fit.id}"

    ## the fit object
    context["fit"] = fit

    ## non-empty fields and values
    # this does nor return a nice rep of foreign keys
    #fields_and_values = [(field.name, field.value_to_string(fit)) for field in modelcls._meta.fields if field.name != "comment" and getattr(fit, field.name) is not None]
    # this does
    fields_and_values = [(field.name, str(getattr(fit, field.name))) for field in modelcls._meta.fields if field.name != "comment" and getattr(fit, field.name) is not None]
    context['fields_and_values'] = fields_and_values

    # If file does not exist, return early

    if not fit.fileexists:
        context["fileexists"] = False
        return  render(request, "iop4admin/view_fitdetails.html", context)
    
    # If file exists, gather additional info and pass it to the template

    context["fileexists"] = True
    context["fitclsname"] = fitclsname
    context["fileloc"] = fit.fileloc
    
    ## header
    with fits.open(fit.filepath) as hdul:
        context["header_L"] = [hdu.header for hdu in hdul]

    ## stats
    context['stats'] = fit.stats
    
    ## urls

    context['url_changelist'] = reverse('iop4admin:%s_%s_changelist' % (modelcls._meta.app_label, modelcls._meta.model_name)) + f"?id={id}"

    ### urls to corresponding raw or reduced fit

    if fitclsname == "RawFit" and hasattr(fit, "reduced"):
            context['url_reduced'] = reverse('iop4admin:%s_%s_changelist' % (modelcls._meta.app_label, fit.reduced._meta.model_name)) + f"?id={fit.reduced.id}"
    if fitclsname == "ReducedFit":
        context['url_raw'] = reverse('iop4admin:%s_%s_changelist' % (modelcls._meta.app_label, fit.rawfit._meta.model_name)) + f"?id={fit.rawfit.id}"

    ### url for link to advance viewer
    context['url_viewer'] = url_viewer= reverse('iop4admin:view_fitviewer', args=[fit.__class__.__name__, fit.id])

    ## image preview parameters and image

    vmin = request.GET.get("vmin", "auto")
    vmax = request.GET.get("vmax", "auto")

    if vmin == "auto":
        vmin = np.quantile(fit.mdata.compressed(), 0.3)
    if vmax == "auto":
        vmax = np.quantile(fit.mdata.compressed(), 0.99)

    imgbytes = fit.get_imgbytes_preview_image(vmin=vmin, vmax=vmax)
    imgb64 = base64.b64encode(imgbytes).decode("utf-8")

    context["vmin"] = vmin
    context["vmax"] = vmax
    context['imgb64'] = imgb64

    # histogram
    histimg = fit.get_imgbytes_preview_histogram()
    histimg_b64 =  base64.b64encode(histimg).decode("utf-8")
    context['histimg_b64'] = histimg_b64
    
    # Reduction info

    ## astrometric calibration
    try:
        context["astrometry_info"] = fit.astrometry_info
    except:
        pass

    astrometry_img_D = dict()
    fnameL = glob.iglob(fit.filedpropdir + "/astrometry_*.png")
    for fname in fnameL:
        with open(fname, "rb") as f:
            imgbytes = f.read()
        imgb64 = base64.b64encode(imgbytes).decode("utf-8")
        astrometry_img_D[os.path.basename(fname)] = imgb64
    context["astrometry_img_D"] = dict(sorted(astrometry_img_D.items()))

    ## sources info
    if fitclsname == "ReducedFit":
        sources_in_field_L = list()
        for source in fit.sources_in_field.all():
            sources_in_field_L.append((source, reverse('iop4admin:%s_%s_changelist' % (source._meta.app_label, source._meta.model_name)) + f"?id={source.id}"))
        context["sources_in_field_L"] = sources_in_field_L

    return  render(request, "iop4admin/view_fitdetails.html", context)