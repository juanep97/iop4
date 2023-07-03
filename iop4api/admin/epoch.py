from django.contrib import admin

from django.utils.html import format_html
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render

from ..filters import *
from ..models import *
from .. import views

class AdminEpoch(admin.ModelAdmin):
    model = Epoch
    list_display = ['id', 'telescope', 'night', 'status', 'count_rawfits', 'count_bias', 'count_flats', 'count_light', 'count_reduced', 'count_calibrated', 'summary_rawfits_status', 'details']
    readonly_fields = [field.name for field in Epoch._meta.fields] 
    ordering = ['-night','-telescope']

    def has_module_permission(self, *args, **kwargs):
        return True
    
    def has_add_permission(self, *args, **kwargs):
        return True

    @admin.display(description='Status')
    def status(self, obj):
        return ", ".join(list(obj.flag_labels))

    @admin.display(description='Total nº of raw files')
    def count_rawfits(self, obj):
        return obj.rawfits.count()
    
    @admin.display(description='Nº of bias')
    def count_bias(self, obj):
        return obj.rawbiascount
    
    @admin.display(description='Nº of flats')
    def count_flats(self, obj):
        return obj.rawflatcount
    
    @admin.display(description='Nº of science files')
    def count_light(self, obj):
        return obj.rawlightcount
    
    @admin.display(description='Nº of reduced files')
    def count_reduced(self, obj):
        return obj.reducedcount
    
    @admin.display(description='Nº of calibrated files')
    def count_calibrated(self, obj):
        return obj.calibratedcount
        
    @admin.display(description='Summary of raw files')
    def summary_rawfits_status(self, obj):
        summary_rawfits_status = obj.get_summary_rawfits_status()
        return summary_rawfits_status
       
    @admin.display(description='Details')
    def details(self, obj):
        url = reverse('iop4admin:view_epochdetails', args=[obj.id])
        url_rawfits = reverse(f'iop4admin:{RawFit._meta.app_label}_{RawFit._meta.model_name}_changelist') + f'?telescope={obj.telescope}&night={obj.night}'
        url_reducedfits = reverse(f'iop4admin:{ReducedFit._meta.app_label}_{ReducedFit._meta.model_name}_changelist')+ f'?telescope={obj.telescope}&night={obj.night}'
        return format_html(rf'<a href="{url}">details</a> / '
                           rf'<a href="{url_rawfits}">rawfits</a> / '
                           rf'<a href="{url_reducedfits}">reducedfits</a>')