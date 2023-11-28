from django.contrib import admin

from django.utils.html import format_html
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render
from django.db.models import Q

# other imports
from iop4api.filters import *
from iop4api.models import *
from iop4lib.enums import *
from .fitfile import AdminFitFile

# logging
import logging
logger = logging.getLogger(__name__)

class AdminReducedFit(AdminFitFile):
    model = ReducedFit
    list_display = ["id", 'filename', 'telescope', 'night', 'instrument', 'status', 'imgtype', 'imgsize', 'band', 'obsmode', 'rotangle', 'exptime', 'get_targets_in_field', 'juliandate', 'options', 'modified']
    readonly_fields = [field.name for field in ReducedFit._meta.fields]
    search_fields = ['id', 'filename', 'epoch__telescope', 'epoch__night', 'sources_in_field__name']
    ordering = ['-epoch__night', '-epoch__telescope', '-juliandate']
    list_filter = (
        RawFitIdFilter,
        RawFitTelescopeFilter,
        RawFitNightFilter,
        RawFitFilenameFilter,
        RawFitInstrumentFilter,
        RawFitFlagFilter,
        "obsmode",
        "band",
        "imgsize",
    )
    
    @admin.display(description='OPTIONS')
    def options(self, obj):
        url_rawfit = reverse('iop4admin:%s_%s_changelist' % (RawFit._meta.app_label, RawFit._meta.model_name)) + f"?id={obj.rawfit.id}"
        url_details = reverse('iop4admin:iop4api_reducedfit_details', args=[obj.id])
        url_viewer= reverse('iop4admin:iop4api_reducedfit_viewer', args=[obj.id])
        return format_html(rf'<a href="{url_rawfit}">raw</a> / <a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')
    
    @admin.display(description='IMGSIZE')
    def imgsize(self, obj):
        return obj.imgsize
    
    @admin.display(description='BAND')
    def band(self, obj):
        return obj.band

    @admin.display(description='OBSMODE')
    def obsmode(self, obj):
        return OBSMODES(obj.obsmode).label
    
    @admin.display(description='ROTANGLE')
    def rotangle(self, obj):
        return obj.rotangle
    
    @admin.display(description='EXPTIME')
    def exptime(self, obj):
        return obj.exptime
    
    @admin.display(description='SRCS IN FIELD')
    def get_targets_in_field(self, obj):

        cat_targets = list(obj.sources_in_field.filter(Q(srctype=SRCTYPES.BLAZAR) | Q(srctype=SRCTYPES.STAR)).values_list('name', flat=True))

        if len(cat_targets) > 0:
            return cat_targets
        
        try:
            kw_obj_val = obj.rawfit.header['OBJECT']
        except FileNotFoundError:
            return format_html(f"<i>rawfit not found</i>")
        
        guessed_target = AstroSource.objects.filter(Q(name__icontains=kw_obj_val) | Q(other_name__icontains=kw_obj_val)).values_list('name', flat=True)

        if len(guessed_target) > 0:
            return format_html(f"<i>guess: {guessed_target[0]} ?</i>") 
        
        return format_html(f"<i>kw: '{kw_obj_val}' ?</i>")