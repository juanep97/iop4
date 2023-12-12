from django.contrib import admin

from django.utils.html import format_html
from django.urls import reverse 
from django.utils.safestring import mark_safe
from iop4api.filters import *
from iop4api.models import *
from .fitfile import AdminFitFile, action_mark_ignore, action_unmark_ignore

import logging
logger = logging.getLogger(__name__)

class AdminMasterBias(AdminFitFile):

    model = MasterBias

    list_display = ['id', 'telescope', 'night', 'instrument', 'imgsize', 'get_built_from', 'options', 'status']

    list_filter = (
            RawFitIdFilter,
            RawFitTelescopeFilter,
            RawFitNightFilter,
            RawFitInstrumentFilter,
            RawFitFlagFilter,
            "imgsize",
    )
    
    actions = [action_mark_ignore, action_unmark_ignore]
    
    @admin.display(description='Options')
    def options(self, obj):
        url_details = reverse(f"iop4admin:{self.model._meta.app_label}_{self.model._meta.model_name}_details", args=[obj.id])
        url_viewer = reverse(f"iop4admin:{self.model._meta.app_label}_{self.model._meta.model_name}_viewer", args=[obj.id])
        return format_html(rf'<a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')
    
    @admin.display(description='Telescope')
    def telescope(self, obj):
        return obj.epoch.telescope
    
    @admin.display(description='Night')
    def night(self, obj):
        return obj.epoch.night
    