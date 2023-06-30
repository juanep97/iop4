from django.urls import path, include

from . import views

from . import admin

from django.conf import settings
from django.conf.urls.static import static

app_name = "iop4api"

urlpatterns = [
    path('', views.index, name='index'),
    ##path('view_fitdetails/<str:fitclsname>/<int:id>/', views.view_fitdetails, name='view_fitdetails') # done from iop4admin views, which provides right permission check
    #path('admin/doc/', include('django.contrib.admindocs.urls')), # adds a link to django built-in documentation for the appliacation, but needs also configuration in settings.py and to have users and groups in the admin
    path("iop4admin/", admin.iop4admin_site.urls),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
