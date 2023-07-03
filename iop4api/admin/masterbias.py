from django.contrib import admin

from django.utils.html import format_html
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render

from ..filters import *
from ..models import *
from .. import views

import logging
logger = logging.getLogger(__name__)

class AdminMasterBias(admin.ModelAdmin):
    model = MasterBias
    list_display = ['id', 'telescope', 'night', 'imgsize', 'options']

    def has_module_permission(self, *args, **kwargs):
        return True
    
    def has_view_permission(self, *args, **kwargs):
        return True
    
    @admin.display(description='Options')
    def options(self, obj):
        url_details = reverse('iop4admin:view_fitdetails', args=["MasterBias", obj.id])
        url_viewer= reverse('iop4admin:view_fitviewer', args=["MasterBias", obj.id])
        return format_html(rf'<a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')
    
    @admin.display(description='Telescope')
    def telescope(self, obj):
        return obj.epoch.telescope
    
    @admin.display(description='Night')
    def night(self, obj):
        return obj.epoch.night