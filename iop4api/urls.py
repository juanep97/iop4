from django.urls import path, include

from . import views

from . import admin

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    ##path('view_fitdetails/<str:fitclsname>/<int:id>/', views.view_fitdetails, name='view_fitdetails') # done from iop4admin views, which provides right permission check
    path('admin/doc/', include('django.contrib.admindocs.urls')),
    path("iop4admin/", admin.iop4admin_site.urls),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
