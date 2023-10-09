import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

from django.db import models
from django.db.models import Q

# other imports
from ..enums import *
import pypandoc 
import warnings
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u

# logging
import logging
logger = logging.getLogger(__name__)

class AstroSourceManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)
    
class AstroSource(models.Model):

    from iop4lib.enums import SRCTYPES

    # DB fields

    # Identification

    name = models.CharField(max_length=255, unique=True)

    # common to all sources

    other_name = models.CharField(max_length=255, null=True, blank=True)
    ra_hms = models.CharField(max_length=255)
    dec_dms = models.CharField(max_length=255)
    srctype = models.CharField(max_length=255, choices=SRCTYPES.choices)  
    comment = models.TextField(null=True, blank=True)

    # Blazar fields
    redshift = models.FloatField(null=True, blank=True)
    
    # Calibration stars fields

    calibrates = models.ManyToManyField('AstroSource', related_name="calibrators", blank=True)

    mag_R = models.FloatField(blank=True, null=True)
    mag_R_err = models.FloatField(blank=True, null=True)

    mag_B = models.FloatField(blank=True, null=True)
    mag_B_err = models.FloatField(blank=True, null=True)

    mag_V = models.FloatField(blank=True, null=True)
    mag_V_err = models.FloatField(blank=True, null=True)

    mag_I = models.FloatField(blank=True, null=True)
    mag_I_err = models.FloatField(blank=True, null=True)

    mag_U = models.FloatField(blank=True, null=True)
    mag_U_err = models.FloatField(blank=True, null=True)


    # Natural key
    # allows us te relate the sources and calibration stars by names only

    # custom manager allows us to use natural keys when loading fixtures
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
                    f" - other_name: {self.other_name}<br>\n"
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

    def get_aperpix(self):
        return 12
    
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

    # Class methods

    @classmethod
    def get_catalog_as_dict(cls, *args, **kwargs):
        objs = cls.objects.filter(*args, **kwargs).all()
        catalog_dict = {obj.id: {'name': obj.name, 
                                 'other_name': obj.other_name,
                                 'srctype': obj.srctype,
                                 'comment': obj.comment,
                                 'coord': obj.coord,
                                 } for obj in objs}
        return catalog_dict
    
    @classmethod
    def get_sources_in_field(cls, wcs=None, height=None, width=None, fit=None):
        if fit is not None:
            wcs = fit.wcs
            height, width = fit.data.shape

        #catalog = cls.get_catalog(Q(srctype=SRCTYPES.BLAZAR) | Q(srctype=SRCTYPES.STAR))
        objs = cls.objects.all()

        sources_in_field = list()

        import warnings
        for obj in objs:
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
