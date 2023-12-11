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
    
    @admin.display(description="Built from")
    def get_built_from(self, obj):
        self.allow_tags = True
        
        ids_str_L = [str(rf.id) for rf in obj.rawfits.all()]
        a_href = reverse('iop4admin:%s_%s_changelist' % (RawFit._meta.app_label, RawFit._meta.model_name)) + "?id__in=%s" % ",".join(ids_str_L)
        a_text = ", ".join(ids_str_L)
        return mark_safe(f'<a href="{a_href}">{a_text}</a>')