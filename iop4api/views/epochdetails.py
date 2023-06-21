from django.shortcuts import render

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.contrib.auth.decorators import login_required
from django.template import loader

from django.contrib.admin.views.decorators import staff_member_required

from ..models import *
from .. import admin

# other imports

import itertools

#@staff_member_required no staff_member_required, because it is only routed from iop4admin_site
def view_epochdetails(sitecls, request, epoch_id):
    
    # add admin context
    context = dict(sitecls.each_context(request))

    # get epoch, return early if it does not exist
    if (epoch := Epoch.objects.filter(id=epoch_id).first()) is None:
        return HttpResponseRedirect(reverse("iop4admin:%s_%s_change" % (Epoch._meta.app_label, Epoch._meta.model_name), args=(epoch_id,)))

    # page title
    context['title'] = f"Epoch {epoch.id}"

    context["epoch"] = epoch

    header_key_S = set(itertools.chain.from_iterable([rawfit.header.keys() for rawfit in epoch.rawfits.all()]))

    #header_key_D = dict()
    #for key in header_key_S:
    #    header_key_D[key] = len([rawfit.id for rawfit in epoch.rawfits.all() if key in rawfit.header])
    
    context["header_key_S"] = list(header_key_S)

    context["rawfitsummarystatus"] = epoch.get_summary_rawfits_status()

    return  render(request, "iop4admin/view_epochdetails.html", context)