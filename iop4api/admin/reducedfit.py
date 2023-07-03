from django.contrib import admin

from django.utils.html import format_html
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render
from django.db.models import Q

from ..filters import *
from ..models import *
from .. import views

# other imports
from iop4lib.enums import *

# logging
import logging
logger = logging.getLogger(__name__)

class AdminReducedFit(admin.ModelAdmin):
    model = ReducedFit
    list_display = ["id", 'filename', 'telescope', 'night', 'status', 'imgtype', 'imgsize', 'band', 'obsmode', 'rotangle', 'exptime', 'get_targets_in_field', 'options', 'modified']
    readonly_fields = [field.name for field in ReducedFit._meta.fields]
    search_fields = ['id', 'filename', 'epoch__telescope', 'epoch__night', 'sources_in_field__name']
    ordering = ['-epoch__night', '-epoch__telescope', '-juliandate']
    list_per_page = 25
    list_filter = (
        RawFitIdFilter,
        RawFitTelescopeFilter,
        RawFitNightFilter,
        RawFitFilenameFilter,
        RawFitFlagFilter,
        "obsmode",
        "band",
        "imgsize",
    )

    def has_module_permission(self, *args, **kwargs):
        return True
    
    def has_view_permission(self, *args, **kwargs):
        return True
    
    @admin.display(description='OPTIONS')
    def options(self, obj):
        #url_rawfit = f"/iop4admin/iop4api/rawfit/?id={obj.id}"
        url_rawfit = reverse('iop4admin:%s_%s_changelist' % (RawFit._meta.app_label, RawFit._meta.model_name)) + f"?id={obj.rawfit.id}"
        url_details = reverse('iop4admin:view_fitdetails', args=["ReducedFit", obj.id])
        url_viewer= reverse('iop4admin:view_fitviewer', args=["ReducedFit", obj.id])
        return format_html(rf'<a href="{url_rawfit}">raw</a> / <a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')
    
    @admin.display(description='TELESCOPE')
    def telescope(self, obj):
        return obj.epoch.telescope
    
    @admin.display(description='NIGHT')
    def night(self, obj):
        return obj.epoch.night
    
    @admin.display(description='FILENAME')
    def filename(self, obj):
        return obj.filename
    
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

    @admin.display(description='STATUS')
    def status(self, obj):
        return ", ".join(obj.flag_labels)
    
    @admin.display(description='SRCS IN FIELD')
    def get_targets_in_field(self, obj):

        cat_targets = list(obj.sources_in_field.filter(Q(srctype=SRCTYPES.BLAZAR) | Q(srctype=SRCTYPES.STAR)).values_list('name', flat=True))

        if len(cat_targets) > 0:
            return cat_targets
        
        kw_obj_val = obj.rawfit.header['OBJECT']
        guessed_target = AstroSource.objects.filter(Q(name__icontains=kw_obj_val) | Q(other_name__icontains=kw_obj_val)).values_list('name', flat=True)

        if len(guessed_target) > 0:
            return format_html(f"<i>guess: {guessed_target[0]} ?</i>") 
        
        return format_html(f"<i>kw: '{kw_obj_val}' ?</i>")