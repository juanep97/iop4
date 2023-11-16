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
    list_display = ['id', 'get_telescope', 'get_instrument', 'get_datetime', 'get_src_name', 'get_src_type', 'get_fwhm', 'get_aperpix', 'get_reducedfit', 'get_obsmode', 'pairs', 'get_rotangle', 'get_src_type', 'get_flux_counts', 'get_flux_counts_err', 'get_bkg_flux_counts', 'get_bkg_flux_counts_err', 'modified']
    readonly_fields = [field.name for field in AperPhotResult._meta.fields]
    search_fields = ['id', 'reducedfit__instrument', 'astrosource__name', 'astrosource__srctype', 'reducedfit__id']
    list_filter = ['reducedfit__instrument', 'astrosource__srctype', 'reducedfit__epoch__telescope', 'reducedfit__obsmode']


    
    @admin.display(description="TELESCOPE")
    def get_telescope(self, obj):
        return obj.reducedfit.epoch.telescope

    @admin.display(description="INSTRUMENT")
    def get_instrument(self, obj):
        return obj.reducedfit.instrument

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
    
    @admin.display(description="fwhm")
    def get_fwhm(self, obj):
        if obj.fwhm is None:
            return "-"
        return f"{obj.fwhm:.1f}"
    
    @admin.display(description="aperpix")
    def get_aperpix(self, obj):
        if obj.aperpix is None:
            return "-"
        return f"{obj.aperpix:.1f}"
    
    @admin.display(description="flux_counts")
    def get_flux_counts(self, obj):
        if obj.flux_counts is None:
            return "-"
        else:
            return f"{obj.flux_counts:.1f}"
    
    @admin.display(description="flux_counts_err")
    def get_flux_counts_err(self, obj):
        if obj.flux_counts_err is None:
            return "-"
        else:
            return f"{obj.flux_counts_err:.1f}"
    
    @admin.display(description="bkg_flux_counts")
    def get_bkg_flux_counts(self, obj):
        if obj.bkg_flux_counts is None:
            return "-"
        else:
            return f"{obj.bkg_flux_counts:.1f}"
    
    @admin.display(description="bkg_flux_counts_err")
    def get_bkg_flux_counts_err(self, obj):
        if obj.bkg_flux_counts_err is None:
            return "-"
        else:
            return f"{obj.bkg_flux_counts_err:.1f}"
    