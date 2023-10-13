from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from . import views

app_name = "iop4api"

class TabsConverter:
    regex = '.*'

    def to_python(self, value):
        return value.split('/')

    def to_url(self, value):
        return '/'.join(value)
    
from django.urls import register_converter
register_converter(TabsConverter, 'tabs')

urlpatterns = [
    path('', views.index, name='index'),
    path('api/login/', views.login_view, name='login_view'),
    path('api/logout/', views.logout_view, name='logout_view'),
    path('api/plot/', views.plot, name='plot'),
    path('api/data/', views.data, name='data'),
    path('api/catalog/', views.catalog, name='catalog'),
    path('api/log/', views.log, name='log'),
    path('api/test/', views.test, name='test'),
    path('<tabs:tabs>/', views.index, name='index'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
