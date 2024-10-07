# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.contrib import admin, messages
from django.utils.html import format_html
from django.urls import path, reverse 
from django.db.models import Avg

# other imports
from iop4api.models import AstroSource, ReducedFit
from iop4admin import views

import numpy as np
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord

# iop4lib imports
from iop4lib.enums import BANDS

# logging
import logging
logger = logging.getLogger(__name__)

class AdminAstroSource(admin.ModelAdmin):
    model = AstroSource
    list_display = ['name', 'other_names', 'ra_hms', 'dec_dms', 'srctype', 'get_last_reducedfit', 'get_last_mag_R', 'get_calibrates', 'get_comment_firstline', 'get_details']
    search_fields = ['name', 'other_names', 'ra_hms', 'dec_dms', 'srctype', 'comment']
    list_filter = ('srctype',)
    actions = ['add_field_stars_from_panstarrs', 'remove_field_stars_from_panstarrs']
    
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
    

    @admin.action(description='Automatically add field stars from PanSTARRS')
    def add_field_stars_from_panstarrs(self, request, queryset):

        if queryset.count() > 1:
            messages.error(request, "Only one source at a time, please.")
            return
        
        main_src = queryset.first()

        logger.info(f"Querying PanSTARRS around {main_src.name} ({main_src.coord.ra.deg} {main_src.coord.dec.deg})")

        try:
            catalog_data = Catalogs.query_criteria(coordinates=f"{main_src.coord.ra.deg} {main_src.coord.dec.deg}", catalog="PANSTARRS", version=1, radius="0.15 deg")
        except Exception as e:
            logger.error(f"Error querying PanSTARRS around {main_src.name} ({main_src.coord.ra.deg} {main_src.coord.dec.deg}): {e}")
            messages.error(request, f"Error querying PanSTARRS around {main_src.name}: {e}")

        logger.info(f"Got {len(catalog_data)} PanSTARRS field stars around {main_src.name}")

        idx = np.isfinite(catalog_data["raStack"])
        idx = idx.data & idx.mask
        catalog_data = catalog_data[idx]

        for col in ["raMean", "decMean", "gMeanPSFMag", "rMeanPSFMag", "iMeanPSFMag", "yMeanPSFMag"]: # "zMeanPSFMag"
            idx = np.isfinite(catalog_data[col])
            idx = idx.data & idx.mask
            catalog_data = catalog_data[~idx]

        idx = (10.5 <= catalog_data["rMeanPSFMag"]) & (catalog_data["rMeanPSFMag"] <= 17.5)
        catalog_data = catalog_data[idx]

        catalog_data.remove_columns([col for col in catalog_data.columns if col not in ["objName", 
                                                                                        "raStack", 
                                                                                        "raMean", "decMean",
                                                                                        "gMeanPSFMag", "gMeanPSFMagErr",
                                                                                        "rMeanPSFMag", "rMeanPSFMagErr", 
                                                                                        "iMeanPSFMag", "iMeanPSFMagErr",
                                                                                        "zMeanPSFMag", "zMeanPSFMagErr",
                                                                                        "yMeanPSFMag", "yMeanPSFMagErr"]])

        logger.info(f"Filtered down to {len(catalog_data)} PanSTARRS field stars for {main_src.name}")

        if len(catalog_data) == 0:
            logger.error(f"No PanSTARRS field stars found for {main_src.name}, skipping")

        field_stars = list()

        for row in catalog_data:
            try:
                cat_ra, cat_dec = row["raMean"], row["decMean"]
                ra_hms, dec_dms = SkyCoord(cat_ra, cat_dec, unit="deg").to_string("hmsdms").split()

                B = row["gMeanPSFMag"] + 0.3130*(row["gMeanPSFMag"] - row["rMeanPSFMag"]) + 0.2271 
                err_B = np.sqrt((1.313*row["gMeanPSFMagErr"])**2+(0.313*row["rMeanPSFMagErr"])**2+0.0107**2)

                V = row["gMeanPSFMag"] - 0.5784*(row["gMeanPSFMag"] - row["rMeanPSFMag"]) - 0.0038  
                err_V = np.sqrt((0.4216*row["gMeanPSFMagErr"])**2+(0.5784*row["rMeanPSFMagErr"])**2+0.0054**2)

                R = row["rMeanPSFMag"] - 0.2936*(row["rMeanPSFMag"] - row["iMeanPSFMag"]) - 0.1439  
                err_R = np.sqrt((0.7064*row["rMeanPSFMagErr"])**2+(0.2936*row["iMeanPSFMagErr"])**2+0.0072**2)

                I = row["rMeanPSFMag"] - 1.2444*(row["rMeanPSFMag"] - row["iMeanPSFMag"]) - 0.3820;  
                err_I = np.sqrt((0.2444*row["rMeanPSFMagErr"])**2+(1.2444*row["iMeanPSFMagErr"])**2+0.0078**2)

                field_star = AstroSource(name=f"PanSTARRS {cat_ra:.5f} {cat_dec:.5f}", 
                                        other_names=row["objName"],
                                        ra_hms=ra_hms, dec_dms=dec_dms,
                                        comment=(f"PanSTARRS field star for {main_src.name}.\n"
                                                f"Autogenerated from PanSTARRS catalog query.\n"
                                                f"SDSS to Johnson-Cousins transformation by Lupton (2005).\n"
                                                f"Field `other_names` corresponds to PanSTARRS `objName`.\n"),
                                        srctype="star",
                                        mag_B=B, mag_B_err=err_B,
                                        mag_V=V, mag_V_err=err_V,
                                        mag_R=R, mag_R_err=err_R,
                                        mag_I=I, mag_I_err=err_I)
                                                    
                field_star.save()

                field_stars.append(field_star)
            except Exception as e:
                logger.error(f"Error with {row['objName']}: {e}")
                continue

        logger.info(f"Added {len(field_stars)} PanSTARRS field stars for {main_src.name}")

        main_src.calibrators.add(*field_stars)

        messages.success(request, f"Added {len(field_stars)} PanSTARRS field stars for {main_src.name}")


    @admin.action(description='Remove all field stars from PanSTARRS')
    def remove_field_stars_from_panstarrs(self, request, queryset):

        if queryset.count() > 1:
            messages.error(request, "Only one source at a time, please.")
            return
        
        main_src = queryset.first()

        field_stars = main_src.calibrators.filter(name__startswith="PanSTARRS")
        field_stars.delete()
        logger.info(f"Removed {len(field_stars)} PanSTARRS field stars for {main_src.name}")

        messages.success(request, f"Removed {len(field_stars)} PanSTARRS field stars for {main_src.name}")