from django.contrib import admin

from django.utils.html import format_html
from django.urls import reverse 
from django.utils.safestring import mark_safe

from iop4api.filters import (
    RawFitIdFilter,
    RawFitTelescopeFilter,
    RawFitNightFilter,
    RawFitInstrumentFilter,
    RawFitFlagFilter,
)

from iop4api.models import (
        MasterBias,
        MasterDark,
        MasterFlat,
)

from .fitfile import (
    AdminFitFile,
    action_mark_ignore,
    action_unmark_ignore,
)

import logging
logger = logging.getLogger(__name__)

class AdminMasterFile(AdminFitFile):

    list_filter = (
            RawFitIdFilter,
            RawFitTelescopeFilter,
            RawFitNightFilter,
            RawFitInstrumentFilter,
            RawFitFlagFilter,
            "imgsize",
    )
    
    actions = [action_mark_ignore, action_unmark_ignore]

    def get_list_display(self, request):

        list_display = ['id', 'telescope', 'night']

        list_display += [field for field in self.model.margs_kwL if field != "epoch"]

        if hasattr(self.model, 'masterbias'):
            list_display += ['get_masterbias']
            
        list_display += ['get_built_from', 'options', 'status']

        if request.session.get('show_image_preview'):
            list_display += ['image_preview']

        return list_display
    
    @admin.display(description='Options')
    def options(self, obj):
        url_details = reverse(f"iop4admin:{self.model._meta.app_label}_{self.model._meta.model_name}_details", args=[obj.id])
        url_viewer = reverse(f"iop4admin:{self.model._meta.app_label}_{self.model._meta.model_name}_viewer", args=[obj.id])
        return format_html(rf'<a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')

    @admin.display(description='MasterBias')
    def get_masterbias(self, obj):
        self.allow_tags = True
        if obj.masterbias is None:
            return "-"
        url = reverse('iop4admin:%s_%s_changelist' % (MasterBias._meta.app_label, MasterBias._meta.model_name)) + f"?id={obj.masterbias.id}"
        return mark_safe(rf'<a href="{url}">{obj.masterbias.id}</a>')
