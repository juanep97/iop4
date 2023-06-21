from django.contrib import admin

from django.utils.html import format_html
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.shortcuts import render

from iop4lib.enums import *
from ..filters import *
from ..models import *

import logging
logger = logging.getLogger(__name__)

class AdminRawFit(admin.ModelAdmin):
    model = RawFit
    list_display = ["id", 'filename', 'telescope', 'night', 'status', 'imgtype', 'imgsize', 'band', 'obsmode', 'rotangle', 'exptime', 'options']
    readonly_fields = [field.name for field in RawFit._meta.fields] 
    search_fields = ['id', 'filename', 'epoch__telescope', 'epoch__night'] 
    ordering = ['-epoch__night','-epoch__telescope']
    list_per_page = 25
    list_filter = (
            RawFitIdFilter,
            RawFitTelescopeFilter,
            RawFitNightFilter,
            RawFitFilenameFilter,
            RawFitFlagFilter,
            "imgtype",
            "obsmode",
            "band",
            "imgsize",
        )
    
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

    def telescope(self, obj):
        return obj.epoch.telescope
    
    def night(self, obj):
        return obj.epoch.night
    
    @admin.display(description='STATUS')
    def status(self, obj):
        return ", ".join(obj.flag_labels)
    
    @admin.display(description='OPTIONS')
    def options(self, obj):
        if obj.imgtype == IMGTYPES.LIGHT:
            url_reduced = reverse('iop4admin:%s_%s_changelist' % (ReducedFit._meta.app_label, ReducedFit._meta.model_name)) + f"?id={obj.reduced.id}"
            url_details = reverse('iop4admin:view_fitdetails', args=["RawFit", obj.id])
            url_viewer= reverse('iop4admin:view_fitviewer', args=["RawFit", obj.id])
            return format_html(rf'<a href="{url_reduced}">reduced</a> / <a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')
        else:
            url_details = reverse('iop4admin:view_fitdetails', args=["RawFit", obj.id])
            url_viewer= reverse('iop4admin:view_fitviewer', args=["RawFit", obj.id])
            return format_html(rf'<a href="{url_details}">details</a> / <a href="{url_viewer}">advanced viewer</a>')

    def image_preview(self, obj, allow_tags=True):
        url_img_preview = reverse('iop4admin:view_fitpreview', args=["RawFit", obj.id])
        return format_html(rf"<img src='{url_img_preview}' width='64' height='64' />")