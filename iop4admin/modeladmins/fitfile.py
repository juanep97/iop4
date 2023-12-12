# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.contrib import admin

from django.utils.html import format_html
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseNotFound, FileResponse
from django.apps import apps
from django.utils.safestring import mark_safe

# iop4lib imports
from iop4api.filters import *
from iop4api.models import *
from iop4admin.views import FitJS9View, FitDetailsView, get_fit_view
from iop4lib.enums import *

# other imports

# logging
import logging
logger = logging.getLogger(__name__)
    

@admin.action(description="Ignore these files in subsequent runs")
def action_mark_ignore(modeladmin, request, queryset):
    for obj in queryset:
        logger.debug(f"Setting IGNORE flag on {obj}")
        obj.set_flag(modeladmin.model.FLAGS.IGNORE)
        obj.save()

@admin.action(description="Do NOT ignore these files in subsequent runs")
def action_unmark_ignore(modeladmin, request, queryset):
    for obj in queryset:
        logger.debug(f"Removing IGNORE flag on {obj}")
        obj.unset_flag(modeladmin.model.FLAGS.IGNORE)
        obj.save()

class AdminFitFile(admin.ModelAdmin):

    list_per_page = 25
    list_max_show_all = 200
    
    @admin.display(description='TELESCOPE', ordering='epoch__telescope')
    def telescope(self, obj):
        return obj.epoch.telescope
    
    @admin.display(description='NIGHT', ordering='epoch__night')
    def night(self, obj):
        return obj.epoch.night
    
    @admin.display(description='FILENAME', ordering='filename')
    def filename(self, obj):
        return obj.filename

    @admin.display(description='STATUS')
    def status(self, obj):
        return ", ".join(obj.flag_labels)
    
    @admin.display(description="Built from")
    def get_built_from(self, obj):
        self.allow_tags = True
        
        ids_str_L = [str(rf.id) for rf in obj.rawfits.all()]
        a_href = reverse('iop4admin:%s_%s_changelist' % (RawFit._meta.app_label, RawFit._meta.model_name)) + "?id__in=%s" % ",".join(ids_str_L)
        a_text = ", ".join(ids_str_L)
        return mark_safe(f'<a href="{a_href}">{a_text}</a>')
    
    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('details/<int:pk>', self.admin_site.admin_view(get_fit_view(FitDetailsView,self.model).as_view()),  name=f"iop4api_{self.model._meta.model_name}_details"),
            path('viewer/<int:pk>', self.admin_site.admin_view(get_fit_view(FitJS9View,self.model).as_view()), name=f"iop4api_{self.model._meta.model_name}_viewer"),
            path('getfile/<int:pk>', self.admin_site.admin_view(self.view_getfile),  name=f"iop4api_{self.model._meta.model_name}_getfile"),
            path('preview/<int:pk>', self.admin_site.admin_view(self.view_preview),  name=f"iop4api_{self.model._meta.model_name}_preview"),
        ]
        return my_urls + urls
    
    def get_absolute_url(self):
        return reverse(f"iop4admin:{self.model._meta.app_label}_{self.model._meta.model_name}_details", args=[self.id])
    
    def view_getfile(self, request, pk):
        if ((fit := self.model.objects.filter(id=pk).first()) is None):
            return HttpResponseNotFound()

        if not fit.fileexists:
            return HttpResponseNotFound()

        return FileResponse(open(fit.filepath, 'rb'), filename=fit.filename)
    
    def view_preview(self, request, pk):
        
        if ((fit := self.model.objects.filter(id=pk).first()) is None):
            return HttpResponseNotFound()

        if not fit.fileexists:
            return HttpResponseNotFound()

        imgbytes = fit.get_imgbytes_preview_image()
        return HttpResponse(imgbytes, content_type="image/png")

    change_list_template = 'iop4admin/fits/change_list.html'
    
    def changelist_view(self, request, extra_context=None):
        # this gets the image preview checkbox value and leaves the url unaltered, otherwise
        # the url would redirect to ?e=1 because of the unexpected parameters
        request.GET._mutable = True
        toggle_image_preview = request.GET.pop('toggle_image_preview', False)
        request.GET._mutable = False

        # toggle the stored session option show_image_preview according to the GET parameter.

        if 'show_image_preview' not in request.session:
            show_image_preview = False
        else:
            show_image_preview = bool(request.session['show_image_preview'])

        if bool(toggle_image_preview):
            show_image_preview = not show_image_preview

        request.session['show_image_preview'] = show_image_preview

        # pass the current filters and page number to template so they get added to the url, otherwise they would be lost when toggling the cbox
        keep_in_url_dict = dict()
        filter_parameter_names = [f"{f}__exact" if isinstance(f,str) else f.parameter_name for f in self.get_list_filter(request)]
        keep_in_url_dict = {k: v for k, v in request.GET.items() if k in filter_parameter_names}
        if 'id' in request.GET: keep_in_url_dict['id'] = request.GET.get('id') # also id
        keep_in_url_dict['p'] = request.GET.get('p', 1) # also page number

        # build extra_context
        extra_context = extra_context or {}
        extra_context['keep_in_url_dict'] = keep_in_url_dict
        extra_context['show_image_preview'] = show_image_preview # will be used to display the cbox as checked or not

        return super().changelist_view(request, extra_context=extra_context)
    
    def get_list_display(self, request):
        list_display = list(super().get_list_display(request))

        if request.session.get('show_image_preview'):
            list_display = self.list_display + ['image_preview']

        return list_display
    
    def image_preview(self, obj, allow_tags=True):
        url_img_preview = reverse(f"iop4admin:iop4api_{self.model._meta.model_name}_preview", args=[obj.id])
        return format_html(rf"<img src='{url_img_preview}' width='64' height='64' />")