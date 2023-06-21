from django.contrib import admin

from django.utils.html import format_html
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render

from ..filters import *
from ..models import *
from .. import views

# IOP4Admin custom admin

class IOP4AdminSite(admin.AdminSite):
    site_title = "IOP4 admin"
    default_site = 'iop4api.admin.IOP4AdminSite'
    site_header = 'IOP4 admin site'
    index_title = 'Welcome to IOP4 admin site'

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path(r'getfile/<str:filecls>/<int:id>/', self.admin_view(self.view_getfile), name='view_getfile'), # API endpoint download a file.
            path(r'fitpreview/<str:fitclsname>/<int:id>/', self.admin_view(self.view_fitpreview), name='view_fitpreview'), # API endpoint to download a preview of a FITS.
            path(r'fitviewer/<str:fitclsname>/<int:id>/', self.admin_view(self.view_fitviewer), name='view_fitviewer'), # a web page to view a fit with JS9
            path(r'fitdetails/<str:fitclsname>/<int:id>/', self.admin_view(self.view_fitdetails), name='view_fitdetails'), # a web page to view details about a FITS.
            path(r'epochdetails/<int:epoch_id>/', self.admin_view(self.view_epochdetails), name='view_epochdetails'), # a web page to view details about an epoch.
            path(r'astrosourcedetails/<int:astrosource_id>/', self.admin_view(self.view_astrosourcedetails), name='view_astrosourcedetails'), # a web page to view details about an astrosource.
        ]
        urls = my_urls + urls
        return urls

    def view_getfile(self, request, *args, **kwargs):
        return views.view_getfile(self, request, *args, **kwargs)
    
    def view_fitviewer(self, request, *args, **kwargs):
        return views.view_fitviewer(self, request, *args, **kwargs)
    
    def view_fitpreview(self, request, *args, **kwargs):
        return views.view_fitpreview(self, request, *args, **kwargs)
    
    def view_fitdetails(self, request, *args, **kwargs):
        return views.view_fitdetails(self, request, *args, **kwargs)
    
    def view_epochdetails(self, request, *args, **kwargs):
        return views.view_epochdetails(self, request, *args, **kwargs)
    
    def view_astrosourcedetails(self, request, *args, **kwargs):
        return views.view_astrosourcedetails(self, request, *args, **kwargs)
    

iop4admin_site = IOP4AdminSite(name='iop4admin')

# Define admin classes.

from .epoch import AdminEpoch
from .rawfit import AdminRawFit
from .masterflat import AdminMasterFlat
from .masterbias import AdminMasterBias
from .reducedfit import AdminReducedFit
from .astrosource import AdminAstroSource
from .aperphotresult import AdminAperPhotResult
from .photopolresult import AdminPhotoPolResult
     
# Register admin classes.

# in default admin site

#admin.site.register(Epoch, AdminEpoch)
#admin.site.register(RawFit, AdminRawFit)
#admin.site.register(Flag, AdminFlag)

# in my custom admin site
# can be done with @admin.register decorator
# like: @admin.register(AstroSource, site=iop4admin_site)

iop4admin_site.register(Epoch, AdminEpoch)
iop4admin_site.register(RawFit, AdminRawFit)
iop4admin_site.register(MasterBias, AdminMasterBias)
iop4admin_site.register(MasterFlat, AdminMasterFlat)
iop4admin_site.register(ReducedFit, AdminReducedFit)
iop4admin_site.register(AstroSource, AdminAstroSource)
iop4admin_site.register(PhotoPolResult, AdminPhotoPolResult)
iop4admin_site.register(AperPhotResult, AdminAperPhotResult)

# from django.contrib import admin
# from django.contrib.auth.models import User, Group

# iop4admin_site.register(User)
# iop4admin_site.register(Group)