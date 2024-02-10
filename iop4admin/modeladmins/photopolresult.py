from django.contrib import admin

from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render

from iop4api.filters import *
from iop4api.models import *

# other imports
from iop4lib.enums import *
from astropy.time import Time

class AdminPhotoPolResult(admin.ModelAdmin):
    model = PhotoPolResult
    list_display = ['id', 'get_telescope', 'get_juliandate', 'get_datetime', 'get_src_name', 'get_src_type', 'get_reducedfits', 'obsmode', 'band', 'exptime', 'get_mag', 'get_mag_err', 'get_p', 'get_p_err', 'get_chi', 'get_chi_err', 'get_aperpix', 'get_aperas', 'get_flags', 'modified']
    readonly_fields = [field.name for field in PhotoPolResult._meta.fields]
    search_fields = ['id', 'astrosource__name', 'astrosource__srctype', 'epoch__night']
    ordering = ['-juliandate']
    list_filter = ['astrosource__srctype', 'epoch__telescope', 'obsmode']


    
    @admin.display(description="TELESCOPE")
    def get_telescope(self, obj):
        return obj.epoch.telescope

    @admin.display(description="DATETIME", ordering='-juliandate')
    def get_datetime(self, obj):
        return Time(obj.juliandate, format='jd').strftime('%Y-%m-%d %H:%M:%S')
    
    @admin.display(description="SRCNAME")
    def get_src_name(self, obj):
        return obj.astrosource.name
    
    @admin.display(description="SRCTYPE")
    def get_src_type(self, obj):
        return obj.astrosource.srctype
    
    @admin.display(description="ReducedFits")
    def get_reducedfits(self, obj):
        self.allow_tags = True
        
        ids_str_L = [str(reducedfit.id) for reducedfit in obj.reducedfits.all()]
        a_href = reverse('iop4admin:%s_%s_changelist' % (ReducedFit._meta.app_label, ReducedFit._meta.model_name)) + "?id__in=%s" % ",".join(ids_str_L)
        a_text = ", ".join(ids_str_L)
        return mark_safe(f'<a href="{a_href}">{a_text}</a>')
    
    @admin.display(description="JD", ordering='-juliandate')
    def get_juliandate(self, obj):
        return f"{obj.juliandate:.6f}"
    
    @admin.display(description="MAG", ordering='-mag')
    def get_mag(self, obj):
        return f"{obj.mag:.3f}" if obj.mag is not None else None
    
    @admin.display(description="MAGERR", ordering='-mag_err')
    def get_mag_err(self, obj):
        return f"{obj.mag_err:.3f}" if obj.mag_err is not None else None
    
    @admin.display(description="P [%]", ordering='-p')
    def get_p(self, obj):
        return f"{100*obj.p:.2f}" if obj.p is not None else None
    
    @admin.display(description="PERR [%]", ordering='-p_err')
    def get_p_err(self, obj):
        return f"{100*obj.p_err:.2f}" if obj.p_err is not None else None
    
    @admin.display(description="CHI [º]", ordering='-chi')
    def get_chi(self, obj):
        return f"{obj.chi:.2f}" if obj.chi is not None else None
    
    @admin.display(description="CHIERR [º]", ordering='-chi_err')
    def get_chi_err(self, obj):
        return f"{obj.chi_err:.2f}" if obj.chi_err is not None else None
    
    @admin.display(description='aperpix', ordering='-aperpix')
    def get_aperpix(self, obj):
        return f"{obj.aperpix:.2f}" if obj.aperpix is not None else None
    
    @admin.display(description='aperas', ordering='-aperas')
    def get_aperas(self, obj):
        return f"{obj.aperas:.2f}" if obj.aperas is not None else None
    
    @admin.display(description='Status')
    def get_flags(self, obj):
        return ", ".join(list(obj.flag_labels))