from django.shortcuts import render

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.template import loader

from django.apps import apps

from ..models import *
from .. import admin

import logging
logger = logging.getLogger(__name__)

def view_fitviewer(site, request, fitclsname, id):  

    try:
        modelcls = apps.get_model('iop4api', fitclsname)
    except LookupError:
        return HttpResponseNotFound()

    if ((fit := modelcls.objects.filter(id=id).first()) is None):
        return HttpResponseRedirect(reverse('iop4admin:%s_%s_change' % (modelcls._meta.app_label, modelcls._meta.model_name), args=(id,)))

    if not fit.fileexists:
        return HttpResponseRedirect(reverse('iop4admin:%s_%s_change' % (modelcls._meta.app_label, modelcls._meta.model_name), args=(id,)))

    context = dict(site.each_context(request))

    import os
    filesize_mb = os.path.getsize(fit.filepath) / 1024 / 1024
    context['filename'] = fit.filename
    context['filesize_mb_str'] = f"{filesize_mb:.1f}"
    context['fileloc'] = fit.fileloc
    context['fitclsname'] = fitclsname

    # urls
    
    # download FITS url
    context['fits_url'] = reverse('iop4admin:view_getfile', args=[fitclsname, fit.id])

    # Changelist URL (table entry)
    #context['url_changelist'] = f"/iop4admin/iop4api/{fitclsname.lower()}/?id={id}"
    context['url_changelist'] = reverse('iop4admin:%s_%s_changelist' % (modelcls._meta.app_label, modelcls._meta.model_name)) + f"?id={fit.id}"

    # URL to associated raw or reduced fit

    if fitclsname == "RawFit": 
        context['url_raw'] = reverse('iop4admin:%s_%s_changelist' % (modelcls._meta.app_label, fit._meta.model_name)) + f"?id={fit.id}"
        if hasattr(fit, "reduced"):
            context['url_reduced'] = reverse('iop4admin:%s_%s_changelist' % (modelcls._meta.app_label, fit.reduced._meta.model_name)) + f"?id={fit.reduced.id}"
    if fitclsname == "ReducedFit":
        context['url_raw'] = reverse('iop4admin:%s_%s_changelist' % (modelcls._meta.app_label, fit.rawfit._meta.model_name)) + f"?id={fit.rawfit.id}"
        context['url_reduced'] = reverse('iop4admin:%s_%s_changelist' % (modelcls._meta.app_label, fit._meta.model_name)) + f"?id={fit.id}"

            

    return  render(request, "iop4admin/view_fitviewer.html", context)