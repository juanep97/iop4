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
    list_display = ['name', 'other_name', 'ra_hms', 'dec_dms', 'srctype', 'get_last_reducedfit', 'get_details']
    search_fields = ['name', 'other_name', 'ra_hms', 'dec_dms', 'srctype', 'comment']
    list_filter = ('srctype',)


    
    def has_add_permission(self, *args, **kwargs):
        return True
    
    def has_change_permission(self, *args, **kwargs):
        return True
    
    def has_delete_permission(self, *args, **kwargs):
        return True
    
    # # by default exclude calibrators and non-polarized stars
    # def get_queryset(self, request):
    #     # Get the original queryset
    #     queryset = super().get_queryset(request)

    #     # Check if the filter is not already applied by checking if the 'is_active' key is not in the request's GET parameters
    #     if 'srctype__exact' not in request.GET:
    #         # Apply the default filter
    #         queryset = queryset.exclude(srctype=SRCTYPES.CALIBRATOR).exclude(srctype=SRCTYPES.UNPOLARIZED_FIELD_STAR)

    #     return queryset
    
    @admin.display(description='CALIBRATES')
    def get_calibrates(self, obj):
        stars_str_L = obj.calibrates.all().values_list('name', flat=True)
        return "\n".join(stars_str_L)
    
    @admin.display(description='CALIBRATORS')
    def get_calibrators(self, obj):
        stars_str_L = obj.calibrators.all().values_list('name', flat=True)
        return "\n".join(stars_str_L)
    
    @admin.display(description='COMMENT')
    def get_comment_html(self, obj):
        return format_html(obj.comment_html) 
    
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
    

