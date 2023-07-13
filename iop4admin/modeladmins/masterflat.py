from django.contrib import admin

from django.utils.html import format_html
from django.urls import reverse 
from django.utils.safestring import mark_safe

from iop4api.filters import *
from iop4api.models import *
from .fitfile import AdminFitFile

import logging
logger = logging.getLogger(__name__)

class AdminMasterFlat(AdminFitFile):
    model = MasterFlat
    list_display = ['id', 'telescope', 'night', 'imgsize', 'band', 'obsmode', 'rotangle', 'exptime', 'masterbias', 'get_built_from', 'options']

    def has_module_permission(self, *args, **kwargs):
        return True
    
    def has_view_permission(self, *args, **kwargs):
        return True
    
    @admin.display(description='Options')
    def options(self, obj):
        url_details = reverse('iop4admin:iop4api_masterflat_details', args=[obj.id])
        url_viewer= reverse('iop4admin:iop4api_masterflat_details', args=[obj.id])
        return format_html(rf'<a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')
    
    @admin.display(description='Telescope')
    def telescope(self, obj):
        return obj.epoch.telescope
    
    @admin.display(description='Night')
    def night(self, obj):
        return obj.epoch.night
    
    @admin.display(description="Built from")
    def get_built_from(self, obj):
        self.allow_tags = True
        link_L = list()
        for rawfit in obj.rawfits.all():
            url = reverse('iop4admin:%s_%s_changelist' % (RawFit._meta.app_label, RawFit._meta.model_name)) + f"?id={rawfit.id}"
            link_L.append(rf'<a href="{url}">{rawfit.id}</a>')
        return mark_safe(", ".join(link_L))