# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.contrib import admin
from django.utils.html import format_html
from django.urls import path, reverse 
from django.db.models import Avg

# other imports
from iop4api.models import AstroSource, ReducedFit
from iop4admin import views

# iop4lib imports
from iop4lib.enums import BANDS

# logging
import logging
logger = logging.getLogger(__name__)

class AdminAstroSource(admin.ModelAdmin):
    model = AstroSource
    list_display = ['name', 'other_name', 'ra_hms', 'dec_dms', 'srctype', 'get_last_reducedfit', 'get_last_mag_R', 'get_calibrates', 'get_comment_firstline', 'get_details']
    search_fields = ['name', 'other_name', 'ra_hms', 'dec_dms', 'srctype', 'comment']
    list_filter = ('srctype',)
    
    @admin.display(description='CALIBRATES')
    def get_calibrates(self, obj):
        stars_str_L = obj.calibrates.all().values_list('name', flat=True)
        return "\n".join(stars_str_L)
    
    @admin.display(description='COMMENTS')
    def get_comment_firstline(self, obj):
        lines = obj.comment.split("\n")
        txt = lines[0]
        if len(lines) > 0 or len(lines[0]) > 30:
            txt = txt[:30] + "..."
        return txt
    
    @admin.display(description='LAST FILE')
    def get_last_reducedfit(self, obj):
        redf = obj.in_reducedfits.order_by('-epoch__night').first()
        if redf is not None:
            url = reverse('iop4admin:%s_%s_changelist' % (ReducedFit._meta.app_label, ReducedFit._meta.model_name)) + "?id=%s" % redf.pk
            return format_html(rf'<a href="{url}">{redf.epoch.night}</a>')
        else:
            return None
    
    @admin.display(description="LAST MAG")
    def get_last_mag_R(self, obj):
        ## get the average of last night
        last_night = obj.photopolresults.filter(band=BANDS.R).earliest('-epoch__night').epoch.night
        r_avg = obj.photopolresults.filter(band=BANDS.R, epoch__night=last_night).aggregate(mag_avg=Avg('mag'), mag_err_avg=Avg('mag_err'))

        mag_r_avg = r_avg.get('mag_avg', None)
        mag_r_err_avg = r_avg.get('mag_err_avg', None)

        if mag_r_avg is not None:
            return f"{mag_r_avg:.2f}"
        else:
            return None
    
    @admin.display(description='DETAILS')
    def get_details(self, obj):
        url = reverse('iop4admin:iop4api_astrosource_details', args=[obj.id])
        return format_html(rf'<a href="{url}">Details</a>')
    
    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('details/<int:pk>', self.admin_site.admin_view(views.AstroSourceDetailsView.as_view()), name='iop4api_astrosource_details'),
        ]
        return my_urls + urls
    

