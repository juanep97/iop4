from django.contrib import admin

from django.utils.html import format_html
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render

from iop4api.filters import *
from iop4api.models import *
from iop4admin import views

import itertools

class AdminEpoch(admin.ModelAdmin):
    model = Epoch
    list_display = ['id', 'telescope', 'night', 'status', 'count_rawfits', 'count_bias', 'count_darks', 'count_flats', 'count_light', 'count_reduced', 'count_calibrated', 'summary_rawfits_status', 'details']
    search_fields = ['id', 'telescope', 'night']
    readonly_fields = [field.name for field in Epoch._meta.fields] 
    ordering = ['-night','-telescope']
    list_filter = ['telescope', 'night']



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
    
    @admin.display(description='Nº of darks')
    def count_darks(self, obj):
        return obj.rawdarkcount
    
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
        url = reverse('iop4admin:iop4api_epoch_details', args=[obj.id])
        url_rawfits = reverse(f'iop4admin:iop4api_rawfit_changelist') + f'?telescope={obj.telescope}&night={obj.night}'
        url_reducedfits = reverse(f'iop4admin:iop4api_reducedfit_changelist')+ f'?telescope={obj.telescope}&night={obj.night}'
        return format_html(rf'<a href="{url}">details</a> / '
                           rf'<a href="{url_rawfits}">rawfits</a> / '
                           rf'<a href="{url_reducedfits}">reducedfits</a>')
    

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('details/<int:pk>', self.admin_site.admin_view(views.EpochDetailsView.as_view()), name='iop4api_epoch_details'),
        ]
        return my_urls + urls
    