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
    list_display = ['name', 'other_names', 'ra_hms', 'dec_dms', 'srctype', 'get_last_reducedfit', 
                    'get_last_mag_R', 'get_calibrates',
                    'get_texp_andor90', 'get_texp_andor150', 'get_texp_dipol', 
                    'get_comment_firstline', 'get_details']
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
        redf = obj.last_reducedfit
        if redf is not None:
            url = reverse('iop4admin:%s_%s_changelist' % (ReducedFit._meta.app_label, ReducedFit._meta.model_name)) + "?id=%s" % redf.pk
            return format_html(rf'<a href="{url}">{redf.epoch.night}</a>')
        else:
            return None
    
    @admin.display(description="LAST MAG")
    def get_last_mag_R(self, obj):
        mag_r_avg, mag_r_err_avg = obj.last_night_mag_R
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
    
    @admin.display(description='T. Andor90')
    def get_texp_andor90(self, obj):
        last_night_mag_R, _ = obj.last_night_mag_R
        texp = obj.texp_andor90

        if last_night_mag_R is None:
            return None
        
        if texp is None:
            return "X"

        return texp
    
    @admin.display(description='T. Andor150')
    def get_texp_andor150(self, obj):
        last_night_mag_R, _ = obj.last_night_mag_R
        texp = obj.texp_andor150

        if last_night_mag_R is None:
            return None
        
        if texp is None:
            return "X"

        return texp
    
    @admin.display(description='T. x N DIPOL')
    def get_texp_dipol(self, obj):
        last_night_mag_R, _ = obj.last_night_mag_R
        texp = obj.texp_dipol
        nreps = obj.nreps_dipol

        if last_night_mag_R is None:
            return None
        
        if texp is None:
            return "X"
        
        return f"{texp} x {nreps}"

    @admin.action(description='Automatically add field stars from PanSTARRS')
    def add_field_stars_from_panstarrs(self, request, queryset):

        if queryset.count() > 1:
            messages.error(request, "Only one source at a time, please.")
            return
        
        main_src = queryset.first()

        # quality constraints
        MIN_R_MAG = 11 # minimum R magnitude
        MAX_R_MAG = 16  # maximum R magnitude
        MAX_MAG_STD = 0.02 # maximum standard deviation in the required bands (we want our calibrators to be stable)
        MIN_N_OBS = 5 # minimum number of observations in the required bands

        logger.info(f"Querying PanSTARRS around {main_src.name} ({main_src.coord.ra.deg} {main_src.coord.dec.deg})")

        try:
            catalog_data = Catalogs.query_criteria(coordinates=f"{main_src.coord.ra.deg} {main_src.coord.dec.deg}", catalog="PANSTARRS", version=1, radius="0.12 deg")
        except Exception as e:
            logger.error(f"Error querying PanSTARRS around {main_src.name} ({main_src.coord.ra.deg} {main_src.coord.dec.deg}): {e}")
            messages.error(request, f"Error querying PanSTARRS around {main_src.name}: {e}")
            return

        logger.info(f"Got {len(catalog_data)} PanSTARRS field stars around {main_src.name}")

        column_names = ['objName', 'raMean', 'decMean',
                        'rMeanApMag', 'rMeanApMagErr', 'rMeanApMagStd', 'rMeanApMagNpt',
                        'gMeanApMag', 'gMeanApMagErr', 'gMeanApMagStd', 'gMeanApMagNpt',
                        'iMeanApMag', 'iMeanApMagErr', 'iMeanApMagStd', 'iMeanApMagNpt',
                        'yMeanApMag', 'yMeanApMagErr', 'yMeanApMagStd', 'yMeanApMagNpt']
        
        for col in column_names:

            if np.issubdtype(catalog_data.dtype[col], np.number):
                idx = np.isfinite(catalog_data[col])
                idx = idx.data & idx.mask
                catalog_data = catalog_data[~idx]

        idx = (MIN_R_MAG <= catalog_data["rMeanApMag"]) & (catalog_data["rMeanApMag"] <= MAX_R_MAG)
        idx = idx & (catalog_data["rMeanApMagStd"] < MAX_MAG_STD)
        idx = idx & (catalog_data["rMeanApMagNpt"] > MIN_N_OBS)
        idx = idx & (catalog_data["gMeanApMagStd"] < MAX_MAG_STD)
        idx = idx & (catalog_data["gMeanApMagNpt"] > MIN_N_OBS)
        idx = idx & (catalog_data["iMeanApMagStd"] < MAX_MAG_STD)
        idx = idx & (catalog_data["iMeanApMagNpt"] > MIN_N_OBS)
        idx = idx & (catalog_data["yMeanApMagStd"] < MAX_MAG_STD)
        idx = idx & (catalog_data["yMeanApMagNpt"] > MIN_N_OBS)
        catalog_data = catalog_data[idx]

        catalog_data.sort(['rMeanApMagStd', 'rMeanApMagErr'])

        catalog_data.remove_columns([col for col in catalog_data.columns if col not in column_names])

        logger.info(f"Filtered down to {len(catalog_data)} PanSTARRS field stars with {MIN_R_MAG} <= R <= {MAX_R_MAG}, std < {MAX_MAG_STD} and n_obs > {MIN_N_OBS}")

        if len(catalog_data) == 0:
            logger.error(f"No PanSTARRS field stars found for {main_src.name}, skipping")

        # sort by number of observations in R, take top 10 only
        if len(catalog_data) > 10:
            logger.info("Keeping only top 10 field stars by number of R observations")
            catalog_data.sort('rMeanApMagNpt')
            catalog_data = catalog_data[-10:]

        field_stars = list()

        for row in catalog_data:
            try:
                cat_ra, cat_dec = row["raMean"], row["decMean"]
                ra_hms, dec_dms = SkyCoord(cat_ra, cat_dec, unit="deg").to_string("hmsdms").split()

                # Transform SDSS to Johnson-Cousins (Lupton 2005)

                B = row["gMeanApMag"] + 0.3130*(row["gMeanApMag"] - row["rMeanApMag"]) + 0.2271 
                err_B = np.sqrt((1.313*row["gMeanApMagErr"])**2+(0.313*row["rMeanApMagErr"])**2+0.0107**2)

                V = row["gMeanApMag"] - 0.5784*(row["gMeanApMag"] - row["rMeanApMag"]) - 0.0038  
                err_V = np.sqrt((0.4216*row["gMeanApMagErr"])**2+(0.5784*row["rMeanApMagErr"])**2+0.0054**2)

                R = row["rMeanApMag"] - 0.2936*(row["rMeanApMag"] - row["iMeanApMag"]) - 0.1439  
                err_R = np.sqrt((0.7064*row["rMeanApMagErr"])**2+(0.2936*row["iMeanApMagErr"])**2+0.0072**2)

                I = row["rMeanApMag"] - 1.2444*(row["rMeanApMag"] - row["iMeanApMag"]) - 0.3820;  
                err_I = np.sqrt((0.2444*row["rMeanApMagErr"])**2+(1.2444*row["iMeanApMagErr"])**2+0.0078**2)

                field_star = AstroSource(name=f"PanSTARRS {cat_ra:.5f} {cat_dec:.5f}", 
                                        other_names=row["objName"],
                                        ra_hms=ra_hms, dec_dms=dec_dms,
                                        comment=(f"PanSTARRS field star for {main_src.name}.\n"
                                                f"Object name: `{row['objName']}`\n"
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
        n_field_stars = len(field_stars) # count before deletion
        field_stars.delete()
        logger.info(f"Removed {n_field_stars} PanSTARRS field stars for {main_src.name}")

        messages.success(request, f"Removed {n_field_stars} PanSTARRS field stars for {main_src.name}")