from django.contrib import admin

from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render

# other imports
from iop4api.filters import *
from iop4api.models import *
from iop4api import views

# other imports
from iop4lib.enums import *
from astropy.time import Time

class AdminAperPhotResult(admin.ModelAdmin):
    model = AperPhotResult
    list_display = ['id', 'get_telescope', 'get_datetime', 'get_src_name', 'get_src_type', 'aperpix', 'get_reducedfit', 'get_obsmode', 'pairs', 'get_rotangle', 'get_src_type', 'flux_counts', 'flux_counts_err', 'bkg_flux_counts', 'bkg_flux_counts_err', 'modified']
    readonly_fields = [field.name for field in AperPhotResult._meta.fields]
    search_fields = ['id', 'astrosource__name', 'astrosource__srctype', 'reducedfit__id']
    list_filter = ['astrosource__srctype', 'reducedfit__epoch__telescope', 'reducedfit__obsmode']

    def has_module_permission(self, *args, **kwargs):
        return True
    
    def has_view_permission(self, *args, **kwargs):
        return True
    
    @admin.display(description="TELESCOPE")
    def get_telescope(self, obj):
        return obj.reducedfit.epoch.telescope

    @admin.display(description="DATETIME")
    def get_datetime(self, obj):
        return Time(obj.reducedfit.juliandate, format='jd').strftime('%Y-%m-%d %H:%M:%S')
    
    @admin.display(description="SRCNAME")
    def get_src_name(self, obj):
        return obj.astrosource.name
    
    @admin.display(description="SRCTYPE")
    def get_src_type(self, obj):
        return obj.astrosource.srctype
    
    @admin.display(description="ReducedFit")
    def get_reducedfit(self, obj):
        self.allow_tags = True
        url = reverse('iop4admin:%s_%s_changelist' % (ReducedFit._meta.app_label, ReducedFit._meta.model_name)) + f"?id={obj.reducedfit.id}"
        return mark_safe(rf'<a href="{url}">{obj.reducedfit.id}</a>')    
    
    @admin.display(description="OBSMODE")
    def get_obsmode(self, obj):
        return obj.reducedfit.obsmode
    
    @admin.display(description="ROTANGLE")
    def get_rotangle(self, obj):
        return obj.reducedfit.rotangle
    
