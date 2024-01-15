# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.db import models
from django.db.models import Count
from django.db.models.signals import m2m_changed
from django.dispatch import receiver

# other imports
import math
import numpy as np
import astropy.units as u
from astropy.time import Time

# iop4lib imports
from iop4lib.enums import *
from .fields import FlagChoices, FlagBitField

# logging
import logging
logger = logging.getLogger(__name__)

class PhotoPolResultReducedFitRelation(models.Model):
    photopolresult = models.ForeignKey('PhotoPolResult', on_delete=models.CASCADE, related_name='reducedfits_relations')
    reducedfit = models.ForeignKey('ReducedFit', on_delete=models.CASCADE, related_name='photopolresults_relations')

    astrosource = models.ForeignKey('AstroSource', null=False, on_delete=models.CASCADE)
    reduction = models.CharField(max_length=20, choices=REDUCTIONMETHODS.choices, help_text="Reduction method used.")

    class Meta:
        app_label = 'iop4api'
        constraints = [
            models.UniqueConstraint(fields=['photopolresult', 'reducedfit', 'astrosource', 'reduction'], name='unique_photopolresultreducedfitrelation')
        ]

class PhotoPolResult(models.Model):
    """ Model for Photo-Polarimetric results, from a set of ReducedFits and for a given AstroSource, produced by a reduction method.

    .. note::
        A Photo Polarimetric will be uniquely identified by the reduced fits it had been built from, the source for which the result is, and the reduction method 
        used. Since the reducedfits are a ManyToManyField, we can not add a unique_together constraint to the combination of fields that is enforced in the DB level 
        (Django does not support this); instead we specify a custom m2m relationship and enforce the constraints there.
    
    .. note::
        Some fields like 'juliandate', 'band', 'exptime', etc are automatically derived from the reducedfits. This is done in two ways:
        - Every time an instance is saved, in PhotoPolResult.save(), through PhotoPolResult.clean().
        - Evert time a reducedfit is added or removed from the reducedfits ManyToManyField, through the m2m_changed signal handler in PhotoPolResult.

        These fields could perhaps be better implemented as properties, but having them as real fields simplifies things a lot in the admin or when filtering.

        The fields 'astrosource' and 'reduction' are derived from the m2m relationship PhotoPolResultReducedFitRelation also.

        None of these fields should be set manually, as they are automatically computed on save.
    """

    # DB fields

    ## identifiers

    reducedfits = models.ManyToManyField('ReducedFit', related_name='photopolresults', through='PhotoPolResultReducedFitRelation', through_fields=('photopolresult', 'reducedfit'))

    ## automaticaly computed when setting reducedfits

    astrosource = models.ForeignKey('AstroSource', null=True, on_delete=models.CASCADE, related_name='photopolresults', editable=False)
    reduction = models.CharField(null=True, max_length=20, choices=REDUCTIONMETHODS.choices, help_text="Reduction method used.", editable=False)
    juliandate = models.FloatField(null=True, help_text='Julian date of observation (mean of julian dates of all reducedfits). Automaticaly computed on save, do not edit.', editable=False)
    juliandate_min = models.FloatField(null=True, help_text='Minimum julian date of observation (min of julian dates of all reducedfits). Automaticaly computed on save, do not edit.', editable=False)
    juliandate_max = models.FloatField(null=True, help_text='Maximum julian date of observation (max of julian dates of all reducedfits). Automaticaly computed on save, do not edit.', editable=False)
    epoch = models.ForeignKey("Epoch", null=True, on_delete=models.CASCADE, help_text="Epoch of the observation.", editable=False)
    instrument = models.CharField(null=True, max_length=20, choices=INSTRUMENTS.choices, default=INSTRUMENTS.NONE, help_text="Instrument used for the observation.", editable=False)
    obsmode = models.CharField(null=True, max_length=20, choices=OBSMODES.choices, default=OBSMODES.NONE, help_text="Whether the observation was photometry or polarimetry.", editable=False)
    band = models.CharField(null=True, max_length=10, choices=BANDS.choices, default=BANDS.NONE, help_text="Band of the observation, as in the filter used (R, V, etc).", editable=False)
    exptime = models.FloatField(null=True, help_text="Exposure time in seconds.")

    ## photo-polarimetric reduction info

    aperpix = models.FloatField(null=True, help_text="Aperture radius in pixels.")
    bkg_flux_counts = models.FloatField(null=True, help_text="Background flux in counts from aperture photometry.")
    bkg_flux_counts_err = models.FloatField(null=True, help_text="Error for bkg_flux_counts.")
    flux_counts = models.FloatField(null=True, help_text="Flux in counts from aperture photometry (background-substracted).")
    flux_counts_err = models.FloatField(null=True, help_text="Error for flux_counts.")
    mag_inst = models.FloatField(null=True, help_text="Instrumental magnitude, computed directly from the flux counts.")
    mag_inst_err = models.FloatField(null=True, help_text="Error for mag_inst.")
    mag_zp = models.FloatField(null=True, help_text="Magnitude zero point (if it is a calibrator with known magnitude, it is the computed zero point from it, if it is not a calibrator, it is the zero point used to compute the magnitude of the source")
    mag_zp_err = models.FloatField(null=True, help_text="Error for mag_zp.")

    ## photo-polarimetric extra-info; e.g. to be used for automatic computation of instrumental polarization

    _x_px = models.FloatField(null=True, help_text="pixel position of the source in the image, x coordinate.")
    _y_px = models.FloatField(null=True, help_text="pixel position of the source in the image, y coordinate.")
    _q_nocorr = models.FloatField(null=True, help_text="value without correction for instrumental polarization!")
    _u_nocorr = models.FloatField(null=True, help_text="value without correction for instrumental polarization!")
    _p_nocorr = models.FloatField(null=True, help_text="value without correction for instrumental polarization!")
    _chi_nocorr = models.FloatField(null=True, help_text="value without correction for instrumental polarization!")

    ## photo-polarimetric results

    mag = models.FloatField(null=True, help_text="Magnitude of the source, result of the reduction.")
    mag_err = models.FloatField(null=True, help_text="Error for mag.")

    p = models.FloatField(null=True, help_text="Polarization of the source [0-1], result of the reduction.")
    p_err = models.FloatField(null=True, help_text="Error for p.")

    chi = models.FloatField(null=True, help_text="Polarization angle of the source [deg], result of the reduction.")
    chi_err = models.FloatField(null=True, help_text="Error for chi.")

    ## host galaxy correction 
    aperas = models.FloatField(null=True, help_text="Aperture radius in arcseconds.")
    mag_corr = models.FloatField(null=True, help_text="Magnitude corrected for host galaxy.")
    mag_corr_err = models.FloatField(null=True, help_text="Error for mag_corr.")
    p_corr = models.FloatField(null=True, help_text="Polarization corrected for host galaxy.")
    p_corr_err = models.FloatField(null=True, help_text="Error for p_corr.")
    used_mag_for_corr = models.FloatField(null=True, help_text="Magnitude used for host galaxy correction (when mag is not available).")
    used_mag_err_for_corr = models.FloatField(null=True, help_text="Error for used_mag_for_corr.")

    ## flags
    
    class FLAGS(FlagChoices):
        BAD_PHOTOMETRY = 1 << 1
        BAD_POLARIMETRY = 1 << 2

    flags = FlagBitField(choices=FLAGS.choices(), default=0, help_text="Flags for the quality of the result.")

    ## extra fields

    modified = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'iop4api'
        verbose_name = 'Photo-Polarimetric Result'
        verbose_name_plural = 'Photo-Polarimetric Results'

    # repr and str

    def __repr__(self):
        return f'{self.__class__.__name__}.objects.get(id={self.id!r})'
    
    def __str__(self):
        return f'<{self.__class__.__name__} {self.id} | {self.obsmode} {self.band} {self.astrosource.name} JD {self.juliandate}>'
    
    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f'{self!r}')
        else:
            with p.group(4, f'<{self.__class__.__name__}(', ')>'):
                p.text(f'id: {self.id}')
                p.breakable()
                p.text(f"reducedfits: {list(self.reducedfits.values_list('id', flat=True))}")
                p.breakable()
                p.text(f'{self.obsmode} {self.band} {self.astrosource.name}')
                p.breakable()
                p.text(f'JD: {self.juliandate:.5f} ({Time(self.juliandate, format="jd").iso})')
                if self.mag is not None:
                    p.breakable()
                    p.text(f'mag: {self.mag:.3f} ± {self.mag_err:.3f}')
                if self.p is not None:
                    p.breakable()
                    p.text(f'p: {self.p:.3f} ± {self.p_err:.3f}')
                if self.chi is not None:
                    p.breakable()
                    p.text(f'chi: {self.chi:.3f} ± {self.chi_err:.3f}')    

    @classmethod
    def qs_exact_reducedfits(cls, reducedfits):
        """ Returns a queryset of PhotoPolResult with exactly the same set of reducedfits as the ones provided."""
        qs = cls.objects.annotate(reducedfits__count=Count('reducedfits')).filter(reducedfits__count=len(reducedfits))
        for reducedfit in reducedfits:
            qs = qs.filter(reducedfits=reducedfit)
        return qs


    @classmethod
    def create(cls, reducedfits, astrosource, reduction, **kwargs):
        """Creates a new instance of PhotoPolResult, for a set of ReducedFit objects, an AstroSource and a reduction method, or updates an existing one.
        
        Parameters
        ----------
        reducedfits : list of ReducedFit
            List of ReducedFit objects to be associated with the PhotoPolResult. They will be stored on the ManyToMany field 'reducedfits'.
        astrosource : AstroSource
            AstroSource object to be associated with the PhotoPolResult. It will be stored as additional information on the ManyToMany relationship with the 'reducedfits'.
        reduction : str
            Reduction method used. It will be stored as additional information on the ManyToMany relationship with the 'reducedfits'.

        Other Parameters
        ----------------
        Any other keyword argument will be set on the instance created, and saved.

        """

        photopolresult_qs = cls.qs_exact_reducedfits(reducedfits).filter(astrosource=astrosource, reduction=reduction)

        # if none found, we create one, else use existing

        if (photopolresult := photopolresult_qs.first()) is None:
            logger.debug(f'Creating DB entry photopolresult for {reducedfits=}, {astrosource=} and {reduction=}')
            photopolresult = cls(**kwargs)
            photopolresult.save() # we need to save it to use the manytomany field
            photopolresult.reducedfits.set(reducedfits, through_defaults={'astrosource': astrosource, 'reduction': reduction}, clear=True)
            photopolresult.save()
        else:
            logger.debug(f'Db entry for photopolresult already exists for {reducedfits=}, {astrosource=} and {reduction=}, using it instead.')

        # set kwargs fields

        for key, value in kwargs.items():
            setattr(photopolresult, key, value)
            photopolresult.save()

        # update automatically filled fields from reducedfits

        photopolresult.update_fields()

        # save and return

        photopolresult.save()

        return photopolresult

    # Methods that automatically compute fields from the reducedfits

    @classmethod 
    def get_juliandate_from_reducedfits(cls, reducedfits):
        return np.mean([reducedfit.juliandate for reducedfit in reducedfits])
    
    @classmethod
    def get_juliandate_min_from_reducedfits(cls, reducedfits):
        return np.min([reducedfit.juliandate for reducedfit in reducedfits])
    
    @classmethod
    def get_juliandate_max_from_reducedfits(cls, reducedfits):
        return np.max([reducedfit.juliandate for reducedfit in reducedfits])
    
    def get_astrosource(self):
        from .astrosource import AstroSource
        return AstroSource.objects.get(pk=self.reducedfits_relations.values_list('astrosource', flat=True).distinct().get())
    
    def get_reduction(self):
        return self.reducedfits_relations.values_list('reduction', flat=True).distinct().get()
    
    def get_juliandate(self):
        return PhotoPolResult.get_juliandate_from_reducedfits(self.reducedfits.all())
    
    def get_juliandate_min(self):
        return PhotoPolResult.get_juliandate_min_from_reducedfits(self.reducedfits.all())
    
    def get_juliandate_max(self):
        return PhotoPolResult.get_juliandate_max_from_reducedfits(self.reducedfits.all())
    
    def get_epoch(self):
        if not all([reducedfit.epoch == self.reducedfits.first().epoch for reducedfit in self.reducedfits.all()]):
            raise Exception('All reducedfits must have the same epoch')
        return self.reducedfits.first().epoch
    
    def get_band(self):
        if not all([reducedfit.band == self.reducedfits.first().band for reducedfit in self.reducedfits.all()]):
            raise Exception('All reducedfits must have the same band')
        return self.reducedfits.first().band
    
    def get_exptime(self):
        if not all([reducedfit.exptime == self.reducedfits.first().exptime for reducedfit in self.reducedfits.all()]):
            raise Exception('All reducedfits must have the same exptime')
        return self.reducedfits.first().exptime
    
    def get_obsmode(self):
        if not all([reducedfit.obsmode == self.reducedfits.first().obsmode for reducedfit in self.reducedfits.all()]):
            raise Exception('All reducedfits must have the same obsmode')
        return self.reducedfits.first().obsmode
    
    def get_instrument(self):
        if not all([reducedfit.instrument == self.reducedfits.first().instrument for reducedfit in self.reducedfits.all()]):
            raise Exception('All reducedfits must have the same instrument')
        return self.reducedfits.first().instrument
    
    def update_fields(self):
        self.astrosource = self.get_astrosource()
        self.reduction = self.get_reduction()
        self.juliandate = self.get_juliandate()
        self.juliandate_min = self.get_juliandate_min()
        self.juliandate_max = self.get_juliandate_max()
        self.epoch = self.get_epoch()
        self.band = self.get_band()
        self.exptime = self.get_exptime()
        self.obsmode = self.get_obsmode()
        self.instrument = self.get_instrument()

    def clean(self):
        if self.id: # .reducedfits can only be set after saving and therefore after having an id
            self.update_fields()
        else:
            pass
    
    def save(self, *args, **kwargs):
        """ Overriden to enforce clean() before saving. See PhotoPolResult.__docstring__ for more info."""
        self.clean()
        super().save(*args, **kwargs)
        
    # Host galaxy correction
        
    class NoHostCorrectionAvailable(Exception):
        pass

    def compute_host_galaxy_correction(self):
        r""" Computes the host galaxy correction and stores it in the appropriate fields in the DB.

        See [1] and the docstring of iop4lib.utils.get_host_correction for more info.

        References
        ----------
        [1] Nilsson, K., “Host galaxy subtraction of TeV candidate BL Lacertae objects”
            Astronomy and Astrophysics, vol. 475, no. 1, pp. 199-207, 2007. 
            doi:10.1051/0004-6361:20077624. 
            URL: https://ui.adsabs.harvard.edu/abs/2007A%26A...475..199N/abstract

        """

        from iop4lib.utils import get_host_correction

        if self.band != BANDS.R:
            raise PhotoPolResult.NoHostCorrectionAvailable('Host galaxy correction only available for R band')
        
        # compute the used aperture in arcseconds

        aperas = self.aperpix * self.reducedfits.first().pixscale.to(u.arcsec / u.pix).value
        
        # get the host galaxy flux for this aperture

        hostcorr_flux, hostcorr_flux_err = get_host_correction(self.astrosource, aperas)

        if hostcorr_flux is None:
            raise PhotoPolResult.NoHostCorrectionAvailable('No host galaxy correction available for this source')

        mag, mag_err, p, p_err = self.mag, self.mag_err, self.p, self.p_err

        # if there is no magnitude in this result, interpolate the lightcurve

        if mag is None:
            from iop4lib.utils import get_column_values
            qs = PhotoPolResult.objects.filter(astrosource=self.astrosource, band=self.band, mag__isnull=False, flags=0).order_by('juliandate')
            vals = get_column_values(qs, column_names=['juliandate', 'mag', 'mag_err'])
            mag, mag_err = np.interp(self.juliandate, vals['juliandate'], vals['mag']), np.interp(self.juliandate, vals['juliandate'], vals['mag_err'])
            used_mag_for_corr = mag
            used_mag_err_for_corr = mag_err
        else:
            used_mag_for_corr, used_mag_err_for_corr = None, None

        # compute the corrected magnitude and polarization
        # it is just a matter of substracting the host galaxy flux from the observed flux

        obsflux = 3.080e6  * 10 ** (-mag*0.4) # mag R to mJy
        obsflux_err = abs( 3.080e6 * 10**(-mag*0.4) * math.log(10) * 0.4 * mag_err )

        flux_corr = obsflux - hostcorr_flux
        flux_corr_err = math.sqrt( obsflux_err**2 + hostcorr_flux_err**2 )

        mag_corr = -2.5 * math.log10( flux_corr / (3.080e6) )
        mag_corr_err = abs( 2.5 * 1 / (flux_corr * math.log(10)) * flux_corr_err )

        if p is None:
            p_corr, p_corr_err = None, None
        else:
            p_corr = p * obsflux / flux_corr 
            p_corr_err = p_corr * math.sqrt( (p_err/p)**2 + (obsflux_err/obsflux)**2 + (flux_corr_err/flux_corr)**2 )

        # store results in DB and return

        self.mag_corr = mag_corr if used_mag_for_corr is None else None # if we used an interpolated magnitude, we don't store the corrected magnitude
        self.mag_corr_err = mag_corr_err if used_mag_for_corr is None else None

        self.p_corr = p_corr
        self.p_corr_err = p_corr_err

        self.aperas = aperas
        
        self.used_mag_for_corr = used_mag_for_corr
        self.used_mag_err_for_corr = used_mag_err_for_corr

        self.save()

        return mag_corr, p_corr, p_corr_err
        

@receiver(m2m_changed, sender=PhotoPolResult.reducedfits.through)
def reducedfits_on_change(sender, instance, action, **kwargs):
    """ Updates automatically filled fields when -after- reducedfits are altered."""
    if action.startswith('post_'):
        if instance.reducedfits.count() > 0:
            instance.update_fields()
