from django.contrib import admin

from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.urls import path, reverse 
from django.http import HttpResponse, HttpResponseNotFound

# iop4api imports
from iop4api.models import (
    AperPhotResult,
    ReducedFit,
)

# other imports
from astropy.time import Time

class AdminAperPhotResult(admin.ModelAdmin):
    model = AperPhotResult
    list_display = ['id', 'get_telescope', 'get_instrument', 'get_datetime', 'get_src_name', 'get_src_type', 'get_fwhm', 'get_aperpix', 'get_r_in', 'get_r_out', 'get_reducedfit', 'get_obsmode', 'pairs', 'get_rotangle', 'get_src_type', 'get_flux_counts', 'get_flux_counts_err', 'get_bkg_flux_counts', 'get_bkg_flux_counts_err', 'get_image_preview', 'modified']
    readonly_fields = [field.name for field in AperPhotResult._meta.fields]
    search_fields = ['id', 'reducedfit__instrument', 'astrosource__name', 'astrosource__srctype', 'reducedfit__id']
    list_filter = ['reducedfit__instrument', 'astrosource__srctype', 'reducedfit__epoch__telescope', 'reducedfit__obsmode']
    ordering = ['pairs', '-id']

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('preview/<int:pk>', self.admin_site.admin_view(self.view_preview),  name=f"iop4api_{self.model._meta.model_name}_preview"),
            path('img_all/<int:pk>', self.admin_site.admin_view(self.view_img_all),  name=f"iop4api_{self.model._meta.model_name}_img_all"),
        ]
        return my_urls + urls
    
    def view_preview(self, request, pk):
        if ((obj := self.model.objects.filter(id=pk).first()) is None):
            return HttpResponseNotFound()
        imgbytes = obj.get_img()
        return HttpResponse(imgbytes, content_type="image/png")
    
    def view_img_all(self, request, pk):
        if ((obj := self.model.objects.filter(id=pk).first()) is None):
            return HttpResponseNotFound()
        imgbytes = obj.get_img_all()
        return HttpResponse(imgbytes, content_type="image/png")
        
    @admin.display(description="TELESCOPE")
    def get_telescope(self, obj):
        return obj.reducedfit.epoch.telescope

    @admin.display(description="INSTRUMENT")
    def get_instrument(self, obj):
        return obj.reducedfit.instrument

    @admin.display(description="DATETIME")
    def get_datetime(self, obj):
        return Time(obj.reducedfit.juliandate, format='jd').strftime('%Y-%m-%d %H:%M:%S')
    
    @admin.display(description="SRCNAME")
    def get_src_name(self, obj):
        return obj.astrosource.name
    
    @admin.display(description="SRCTYPE")
    def get_src_type(self, obj):
        return obj.astrosource.srctype
    
    @admin.display(description="ReducedFit")
    def get_reducedfit(self, obj):
        self.allow_tags = True
        url = reverse('iop4admin:%s_%s_changelist' % (ReducedFit._meta.app_label, ReducedFit._meta.model_name)) + f"?id={obj.reducedfit.id}"
        return mark_safe(rf'<a href="{url}">{obj.reducedfit.id}</a>')    
    
    @admin.display(description="OBSMODE")
    def get_obsmode(self, obj):
        return obj.reducedfit.obsmode
    
    @admin.display(description="ROTANGLE")
    def get_rotangle(self, obj):
        return obj.reducedfit.rotangle
    
    @admin.display(description="fwhm [as]")
    def get_fwhm(self, obj):
        if obj.fwhm is None:
            return "-"
        return f"{obj.fwhm:.1f}"
    
    @admin.display(description="r_ap [px]")
    def get_aperpix(self, obj):
        if obj.aperpix is None:
            return "-"
        return f"{obj.aperpix:.1f}"
    
    @admin.display(description="r_in [px]")
    def get_r_in(self, obj):
        if obj.r_in is None:
            return "-"
        return f"{obj.r_in:.1f}"
    
    @admin.display(description="r_out [px]")
    def get_r_out(self, obj):
        if obj.r_out is None:
            return "-"
        return f"{obj.r_out:.1f}"
    
    @admin.display(description="flux_counts")
    def get_flux_counts(self, obj):
        if obj.flux_counts is None:
            return "-"
        else:
            return f"{obj.flux_counts:.1f}"
    
    @admin.display(description="flux_counts_err")
    def get_flux_counts_err(self, obj):
        if obj.flux_counts_err is None:
            return "-"
        else:
            return f"{obj.flux_counts_err:.1f}"
    
    @admin.display(description="bkg_flux_counts")
    def get_bkg_flux_counts(self, obj):
        if obj.bkg_flux_counts is None:
            return "-"
        else:
            return f"{obj.bkg_flux_counts:.1f}"
    
    @admin.display(description="bkg_flux_counts_err")
    def get_bkg_flux_counts_err(self, obj):
        if obj.bkg_flux_counts_err is None:
            return "-"
        else:
            return f"{obj.bkg_flux_counts_err:.1f}"
    
    @admin.display(description="img")
    def get_image_preview(self, obj, allow_tags=True):
        url_img_preview = reverse(f"iop4admin:iop4api_{self.model._meta.model_name}_preview", args=[obj.id])
        url_img_all = reverse(f"iop4admin:iop4api_{self.model._meta.model_name}_img_all", args=[obj.id])
        return format_html(rf"<a href='{url_img_all}' target='_blank'><img src='{url_img_preview}' width='64' height='64'/></a>")
