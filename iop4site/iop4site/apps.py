
# define new default admin config
from django.contrib import admin
from django.contrib.admin.apps import AdminConfig

class IOP4AdminConfig(admin.apps.AdminConfig):
    default_site = 'iop4api.admin.IOP4AdminSite'
    site_title = "IOP4 admin"
    site_header = 'IOP4 admin site'