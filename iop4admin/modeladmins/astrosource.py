# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.contrib import admin
from django.utils.html import format_html
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render, get_object_or_404
from django.http import QueryDict

# other imports
from iop4api.filters import *
from iop4api.models import *
from iop4admin import views

# iop4lib imports
from iop4lib.enums import *

# logging
import logging
logger = logging.getLogger(__name__)

class AdminAstroSource(admin.ModelAdmin):
    model = AstroSource
    list_display = ['name', 'other_name', 'ra_hms', 'dec_dms', 'srctype', 'get_last_reducedfit', 'get_calibrates', 'get_comment_firstline', 'get_details']
    search_fields = ['name', 'other_name', 'ra_hms', 'dec_dms', 'srctype', 'comment']
    list_filter = ('srctype',)
    
    @admin.display(description='CALIBRATES')
    def get_calibrates(self, obj):
        stars_str_L = obj.calibrates.all().values_list('name', flat=True)
        return "\n".join(stars_str_L)
    
    @admin.display(description='COMMENTS')
    def get_comment_firstline(self, obj):
        lines = obj.comment.split("\n")
        txt = lines[0]
        if len(lines) > 0 or len(lines[0]) > 30:
            txt = txt[:30] + "..."
        return txt
    
    @admin.display(description='LAST REDUCEDFIT')
    def get_last_reducedfit(self, obj):
        return obj.in_reducedfits.order_by('epoch__night').values_list('epoch__night', flat=True).last()
    
    @admin.display(description='DETAILS')
    def get_details(self, obj):
        url = reverse('iop4admin:iop4api_astrosource_details', args=[obj.id])
        return format_html(rf'<a href="{url}">Details</a>')
    
    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('details/<int:pk>', self.admin_site.admin_view(views.AstroSourceDetailsView.as_view()), name='iop4api_astrosource_details'),
        ]
        return my_urls + urls
    

