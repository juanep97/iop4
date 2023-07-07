import django.contrib.admin.apps

class IOP4AdminConfig(django.contrib.admin.apps.SimpleAdminConfig):
    name = 'iop4admin'
    default_site = 'iop4admin.sites.IOP4AdminSite'