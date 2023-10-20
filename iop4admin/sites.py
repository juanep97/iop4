from django.contrib import admin
from iop4api.filters import *
from iop4api.models import *

# IOP4Admin custom admin

class IOP4AdminSite(admin.AdminSite):
    site_title = "IOP4 admin"
    site_header = 'IOP4 admin site'
    site_url = "/iop4/"
    index_title = 'Welcome to IOP4 admin site'
    
iop4admin_site = IOP4AdminSite(name='iop4admin')

# Register IOP4 models in the admin.

from .modeladmins.epoch import AdminEpoch
from .modeladmins.rawfit import AdminRawFit
from .modeladmins.masterflat import AdminMasterFlat
from .modeladmins.masterdark import AdminMasterDark
from .modeladmins.masterbias import AdminMasterBias
from .modeladmins.reducedfit import AdminReducedFit
from .modeladmins.astrosource import AdminAstroSource
from .modeladmins.aperphotresult import AdminAperPhotResult
from .modeladmins.photopolresult import AdminPhotoPolResult

iop4admin_site.register(Epoch, AdminEpoch)
iop4admin_site.register(RawFit, AdminRawFit)
iop4admin_site.register(MasterBias, AdminMasterBias)
iop4admin_site.register(MasterFlat, AdminMasterFlat)
iop4admin_site.register(MasterDark, AdminMasterDark)
iop4admin_site.register(ReducedFit, AdminReducedFit)
iop4admin_site.register(AstroSource, AdminAstroSource)
iop4admin_site.register(PhotoPolResult, AdminPhotoPolResult)
iop4admin_site.register(AperPhotResult, AdminAperPhotResult)

