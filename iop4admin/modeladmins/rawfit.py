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
    list_display = ["id", 'filename', 'telescope', 'night', 'status', 'imgtype', 'imgsize', 'band', 'obsmode', 'rotangle', 'exptime', 'options']
    readonly_fields = [field.name for field in RawFit._meta.fields] 
    search_fields = ['id', 'filename', 'epoch__telescope', 'epoch__night'] 
    ordering = ['-epoch__night','-epoch__telescope']
    list_per_page = 25
    list_filter = (
            RawFitIdFilter,
            RawFitTelescopeFilter,
            RawFitNightFilter,
            RawFitFilenameFilter,
            RawFitFlagFilter,
            "imgtype",
            "obsmode",
            "band",
            "imgsize",
        )
    
    def has_module_permission(self, *args, **kwargs):
        return True
    
    def has_view_permission(self, *args, **kwargs):
        return True
    
    def telescope(self, obj):
        return obj.epoch.telescope
    
    def night(self, obj):
        return obj.epoch.night
    
    @admin.display(description='STATUS')
    def status(self, obj):
        return ", ".join(obj.flag_labels)
    
    @admin.display(description='OPTIONS')
    def options(self, obj):
        if obj.imgtype == IMGTYPES.LIGHT:
            url_reduced = reverse('iop4admin:%s_%s_changelist' % (ReducedFit._meta.app_label, ReducedFit._meta.model_name)) + f"?id={obj.reduced.id}"
            url_details = reverse('iop4admin:iop4api_rawfit_details', args=[obj.id])
            url_viewer= reverse('iop4admin:iop4api_rawfit_viewer', args=[obj.id])
            return format_html(rf'<a href="{url_reduced}">reduced</a> / <a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')
        else:
            url_details = reverse('iop4admin:iop4api_rawfit_details', args=[obj.id])
            url_viewer= reverse('iop4admin:iop4api_rawfit_viewer', args=[obj.id])
            return format_html(rf'<a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')

    def image_preview(self, obj, allow_tags=True):
        url_img_preview = reverse('iop4admin:iop4api_rawfit_preview', args=[obj.id])
        return format_html(rf"<img src='{url_img_preview}' width='64' height='64' />")