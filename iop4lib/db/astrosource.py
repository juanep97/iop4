import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.db import models
from django.db.models import Exists, OuterRef
from django.db.models import Q, Avg

# iop4lib imports
from ..enums import *

# other imports
import os
import pypandoc 
import warnings
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
import math

# logging
import logging
logger = logging.getLogger(__name__)

class AstroSourceQuerySet(models.QuerySet):
    def with_is_calibrator(self):
        calibrator_subquery = self.filter(calibrators=OuterRef('pk'))
        return self.annotate(
            is_calibrator=Exists(calibrator_subquery)
        )

class AstroSourceManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)
    def get_queryset(self):
        return AstroSourceQuerySet(self.model).with_is_calibrator()
      
class AstroSource(models.Model):
    """ AstroSource model, representing an astronomical source in the IOP4 catalog.

    The coordinates (ra, dec) are interpreted as ICRS coordinates, following astropy `SkyCoord` default convention.
    """

    from iop4lib.enums import SRCTYPES

    # DB fields

    # Identification

    name = models.CharField(max_length=255, unique=True, help_text="Name of the source (must be unique)")

    # common to all sources

    other_names = models.CharField(max_length=255, null=True, blank=True, help_text="Alternative names for the source, separated by a semicolon ';'. It might be needed to correctly identify the target source of observations if observers used a different name.")
    ra_hms = models.CharField(max_length=255, help_text="Right ascension (ICRS) in hh:mm:ss format")
    dec_dms = models.CharField(max_length=255, help_text="Declination (ICRS) in dd:mm:ss format")
    srctype = models.CharField(max_length=255, choices=SRCTYPES.choices, help_text="Source type")  
    comment = models.TextField(null=True, blank=True, help_text="Any comment about the source (in Markdown format)")

    # Blazar fields

    redshift = models.FloatField(null=True, blank=True, help_text="Redshift of the source")
    
    # Calibration stars fields

    calibrates = models.ManyToManyField('AstroSource', related_name="calibrators", blank=True, help_text="sources that it calibrates (for calibrators only)")

    p = models.FloatField(blank=True, null=True, help_text="Polarization degree [0-1] (for calibrators only)")
    p_err = models.FloatField(blank=True, null=True, help_text="Polarization degree error (for calibrators only)")
    
    chi = models.FloatField(blank=True, null=True, help_text="Polarization angle [deg] (for calibrators only)")
    chi_err = models.FloatField(blank=True, null=True, help_text="Polarization angle error (for calibrators only)")

    mag_R = models.FloatField(blank=True, null=True, help_text="Literature magnitude in R band (for calibrators only)")
    mag_R_err = models.FloatField(blank=True, null=True, help_text="Literature magnitude error in R band (for calibrators only)")

    mag_B = models.FloatField(blank=True, null=True, help_text="Literature magnitude in B band (for calibrators only)")
    mag_B_err = models.FloatField(blank=True, null=True, help_text="Literature magnitude error in B band (for calibrators only)")

    mag_V = models.FloatField(blank=True, null=True, help_text="Literature magnitude in V band (for calibrators only)")
    mag_V_err = models.FloatField(blank=True, null=True, help_text="Literature magnitude error in V band (for calibrators only)")

    mag_I = models.FloatField(blank=True, null=True, help_text="Literature magnitude in I band (for calibrators only)")
    mag_I_err = models.FloatField(blank=True, null=True, help_text="Literature magnitude error in I band (for calibrators only)")

    mag_U = models.FloatField(blank=True, null=True, help_text="Literature magnitude in U band (for calibrators only)")
    mag_U_err = models.FloatField(blank=True, null=True, help_text="Literature magnitude error in U band (for calibrators only)")


    # Natural key
    # allows us to relate the sources and calibration stars by names only

    # custom manager allows us to use natural keys when loading fixtures
    # and introduces virtual calibrator boolean field
    objects = AstroSourceManager()
    
    # this method allows us to dump using natural keys
    def natural_key(self):
            return (self.name,)
    
    class Meta:
        app_label = 'iop4api'
        verbose_name = "AstroSource"
        verbose_name_plural = "AstroSources"
    
    # repr and str

    def __repr__(self):
        return f"AstroSource.objects.get(name={self.name!r})"

    def __str__(self):
        return f"<AstroSource {self.name}>"
    
    def _repr_html_(self):
        lvl = logging.getLogger().level
        logging.getLogger().setLevel(logging.CRITICAL)
        res_html =  (f"{self.__class__.__name__}(name={self.name!r}):<br>\n"
                    f" - other_names: {self.other_names}<br>\n"
                    f" - ra={self.coord.ra:.4f}, dec={self.coord.dec:.4f}<br>\n"
                    f" - srctype: {self.srctype}<br>\n"
                    f" - comment: {self.comment_html}<br>\n")
        logging.getLogger().setLevel(lvl)
        return res_html
    
    # methods

    def is_in_field(self, wcs, height, width):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, y = self.coord.to_pixel(wcs)
        except:
            return False
        else:
            if (0 <= x < height) and (0 <= y < width):
                return True
            else:
                return False
            
    # helper properties

    @property
    def other_names_list(self):
        if not self.other_names:
            return []
        elif ';' not in self.other_names:
            return [self.other_names]
        else:
            return [s.strip() for s in self.other_names.split(';')]
        
    @property
    def all_names_list(self):
        return [self.name] + self.other_names_list

    
    @property
    def coord(self):
        return SkyCoord(self.ra_hms, self.dec_dms, unit=(u.hourangle, u.deg))
    
    @property
    def comment_html(self):
        if '<br>' in self.comment:
             fmt = 'html'
             html_src = self.comment
        else:
             fmt = 'md'
             html_src = pypandoc.convert_text(self.comment, 'html', format=fmt, verify_format=False)

        return html_src

    @property
    def filedpropdir(self):
        return os.path.join(iop4conf.datadir, "astrosource", self.name)
    
    # Class methods

    @classmethod
    def get_catalog_as_dict(cls, *args, **kwargs):
        objs = cls.objects.filter(*args, **kwargs).all()
        catalog_dict = {obj.id: {'name': obj.name, 
                                 'other_names': obj.other_names,
                                 'srctype': obj.srctype,
                                 'comment': obj.comment,
                                 'coord': obj.coord,
                                 } for obj in objs}
        return catalog_dict
    
    @classmethod
    def get_sources_in_field(cls, wcs=None, width=None, height=None, fit=None, qs=None):
        r""" Get the sources in the field of view of the image.

        It accepts either a fit image or a wcs, height and width.
        If no query set is given, it will search the whole catalog,
        otherwise it will search the given query set.
        """

        if fit is not None:
            wcs = fit.wcs
            width, height = fit.width, fit.height

        if qs is None:
            qs = cls.objects.all()

        sources_in_field = list()

        import warnings
        for obj in qs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x, y = obj.coord.to_pixel(wcs)
            except:
                pass
            else:
                if (0 <= x < height) and (0 <= y < width):
                    sources_in_field.append(obj)

        return sources_in_field


    @property
    def last_reducedfit(self):
        """Returns the last ReducedFit object associated with the source."""
        return self.in_reducedfits.order_by('-epoch__night').first()
    
    @property
    def last_night_mag_R(self):
        """Returns the average magnitude and error of the last night in the R band."""

        last_night = self.photopolresults.filter(band=BANDS.R).earliest('-epoch__night').epoch.night
        r_avg = self.photopolresults.filter(band=BANDS.R, epoch__night=last_night).aggregate(mag_avg=Avg('mag'), mag_err_avg=Avg('mag_err'))

        mag_r_avg = r_avg.get('mag_avg', None)
        mag_r_err_avg = r_avg.get('mag_err_avg', None)

        return mag_r_avg, mag_r_err_avg
    
    @property
    def texp_andor90(self):
        """Recommneded exposure time for Andor90, based on the last R magnitude and for a SNR of 150."""

        snr = 150
        last_night_mag_R, _ = self.last_night_mag_R

        if last_night_mag_R is None:
            return None

        texp = math.pow(snr,2) * 9.77 * 1e-16 * math.pow(10, 0.8*last_night_mag_R)

        if texp < 30:
            return 60
        elif texp <= 100:
            return 150
        elif texp <= 250:
            return 300
        elif texp <= 400:
            return 450
        elif texp <= 800:
            return 600
        else:
            return None
        
    @property
    def texp_andor150(self):
        """Recommneded exposure time for Andor150, based on the last night R magnitude and for a SNR of 150."""

        snr = 150
        last_night_mag_R, _ = self.last_night_mag_R

        if last_night_mag_R is None:
            return None
        
        texp = 0.36 * math.pow(snr,2) * 9.77 * 1e-16 * math.pow(10, 0.8*last_night_mag_R)

        if texp < 30:
            return 60
        elif texp <= 100:
            return 150
        elif texp <= 250:
            return 300
        elif texp <= 400:
            return 450
        else:
            return 600
    
    @property
    def texp_dipol(self):
        """Recommneded exposure time for DIPOL, based on the last night R magnitude and for a SNR of 150."""

        snr = 150
        last_night_mag_R, _ = self.last_night_mag_R

        if last_night_mag_R is None:
            return None
        
        texp = math.pow(snr,2) * 9.77 * 1e-16 * math.pow(10, 0.8*last_night_mag_R)

        if  texp <= 300:
            return math.ceil(texp / 10) * 10 + 10
        elif texp <= 2000:
            return 300
        else:
            return None
        
    @property
    def nreps_dipol(self):
        """Recommneded number of repetitions for DIPOL, based on the last night R magnitude and for a SNR of 150."""

        snr = 150
        last_night_mag_R, _ = self.last_night_mag_R

        if last_night_mag_R is None:
            return None
        
        texp = math.pow(snr,2) * 9.77 * 1e-16 * math.pow(10, 0.8*last_night_mag_R)

        if texp <= 20:
            return 8
        elif texp <= 40:
            return 4
        elif texp <= 80:
            return 2
        else:
            return 1
