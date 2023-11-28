from django.contrib import admin

from django.utils.html import format_html
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render

from iop4api.filters import *
from iop4api.models import *
from .fitfile import AdminFitFile

import logging
logger = logging.getLogger(__name__)

class AdminRawFit(AdminFitFile):
    model = RawFit
    list_display = ["id", 'filename', 'telescope', 'night', 'instrument', 'status', 'imgtype', 'imgsize', 'band', 'obsmode', 'rotangle', 'exptime', 'options']
    readonly_fields = [field.name for field in RawFit._meta.fields] 
    search_fields = ['id', 'filename', 'epoch__telescope', 'epoch__night'] 
    ordering = ['-epoch__night','-epoch__telescope']
    list_filter = (
            RawFitIdFilter,
            RawFitTelescopeFilter,
            RawFitNightFilter,
            RawFitFilenameFilter,
            RawFitInstrumentFilter,
            RawFitFlagFilter,
            "imgtype",
            "obsmode",
            "band",
            "imgsize",
        )
    
    
    @admin.display(description='OPTIONS')
    def options(self, obj):
        html_src = str()

        if obj.imgtype == IMGTYPES.LIGHT and hasattr(obj, "reduced"):
            url_reduced = reverse('iop4admin:%s_%s_changelist' % (ReducedFit._meta.app_label, ReducedFit._meta.model_name)) + f"?id={obj.reduced.id}"
            html_src += rf'<a href="{url_reduced}">reduced</a> / '
        
        url_details = reverse('iop4admin:iop4api_rawfit_details', args=[obj.id])
        url_viewer= reverse('iop4admin:iop4api_rawfit_viewer', args=[obj.id])
        html_src += rf'<a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>'

        return format_html(html_src)

    def image_preview(self, obj, allow_tags=True):
        url_img_preview = reverse('iop4admin:iop4api_rawfit_preview', args=[obj.id])
        return format_html(rf"<img src='{url_img_preview}' width='64' height='64' />")