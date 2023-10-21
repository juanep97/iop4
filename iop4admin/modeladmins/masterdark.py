from django.contrib import admin

from django.utils.html import format_html
from django.urls import reverse 
from django.utils.safestring import mark_safe

from iop4api.filters import *
from iop4api.models import *
from .fitfile import AdminFitFile

import logging
logger = logging.getLogger(__name__)

class AdminMasterDark(AdminFitFile):
    
    model = MasterDark

    list_display = ['id', 'telescope', 'night', 'instrument', 'imgsize', 'exptime', 'get_masterbias', 'get_built_from', 'options']
    
    list_filter = (
            RawFitIdFilter,
            RawFitTelescopeFilter,
            RawFitNightFilter,
            RawFitInstrumentFilter,
            RawFitFlagFilter,
            "imgsize",
    )

    @admin.display(description='Options')
    def options(self, obj):
        url_details = reverse('iop4admin:iop4api_masterdark_details', args=[obj.id])
        url_viewer= reverse('iop4admin:iop4api_masterdark_details', args=[obj.id])
        return format_html(rf'<a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')
    
    @admin.display(description='Telescope')
    def telescope(self, obj):
        return obj.epoch.telescope
    
    @admin.display(description='Night')
    def night(self, obj):
        return obj.epoch.night
    
    @admin.display(description='MasterBias')
    def get_masterbias(self, obj):
        self.allow_tags = True
        if obj.masterbias is None:
            return "-"
        url = reverse('iop4admin:%s_%s_changelist' % (MasterBias._meta.app_label, MasterBias._meta.model_name)) + f"?id={obj.masterbias.id}"
        return mark_safe(rf'<a href="{url}">{obj.masterbias.id}</a>')
    
    @admin.display(description="Built from")
    def get_built_from(self, obj):
        self.allow_tags = True
        link_L = list()
        for rawfit in obj.rawfits.all():
            url = reverse('iop4admin:%s_%s_changelist' % (RawFit._meta.app_label, RawFit._meta.model_name)) + f"?id={rawfit.id}"
            link_L.append(rf'<a href="{url}">{rawfit.id}</a>')
        return mark_safe(", ".join(link_L))