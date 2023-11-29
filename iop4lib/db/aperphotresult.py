# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

 # django imports
from django.db import models

# other imports

# iop4lib imports
from ..enums import *

# logging
import logging
logger = logging.getLogger(__name__)



class AperPhotResult(models.Model):
    """ Aperture photometry results for an AstroSource in a ReducedFit, either the Ordinary or Extraordinary pair."""

    # database fields

    ## identifiers

    reducedfit = models.ForeignKey("ReducedFit", on_delete=models.CASCADE, related_name='aperphotresults', help_text="The ReducedFit this AperPhotResult has been computed for.")
    astrosource = models.ForeignKey("AstroSource", on_delete=models.CASCADE, related_name='aperphotresults', help_text="The AstroSource this AperPhotResult has been computed for.")
    aperpix = models.FloatField(null=True, blank=True)
    r_in = models.FloatField(null=True, blank=True)
    r_out = models.FloatField(null=True, blank=True)
    pairs = models.TextField(null=True, blank=True, choices=PAIRS.choices, help_text="Whether this AperPhotResult is for the Ordinary or Extraordinary pair.")

    ## photometry results

    bkg_flux_counts = models.FloatField(null=True, blank=True)
    bkg_flux_counts_err = models.FloatField(null=True, blank=True)
    
    flux_counts = models.FloatField(null=True, blank=True)
    flux_counts_err = models.FloatField(null=True, blank=True)

    ## extra fields
    fwhm = models.FloatField(null=True, blank=True)
    photopolresults = models.ManyToManyField("PhotoPolResult", related_name='aperphotresults', help_text="The PhotoPolResult(s) this AperPhotResult has been used for.")
    modified = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'iop4api'
        verbose_name = 'Aperture Photometry Result'
        verbose_name_plural = 'Aperture Photometry Results'
        constraints = [
            models.UniqueConstraint(fields=['reducedfit', 'astrosource', 'aperpix', 'r_in', 'r_out', 'pairs'], name='unique_aperphotresult')
        ]

  # repr and str

    def __repr__(self):
        return f'{self.__class__.__name__}.objects.get(id={self.id!r})'
    
    def __str__(self):
        return f'<{self.__class__.__name__} {self.id} | {self.reducedfit.fileloc} {self.astrosource.name} {self.aperpix} px {self.pairs}>' 

    @classmethod
    def create(cls, reducedfit, astrosource, aperpix, pairs, **kwargs):

        if (result := AperPhotResult.objects.filter(reducedfit=reducedfit, astrosource=astrosource, aperpix=aperpix, pairs=pairs).first()) is not None:
            logger.debug(f"AperPhotResult for {reducedfit}, {astrosource}, {aperpix}, {pairs} already exists, it will be used instead.")
        else:
            logger.debug(f"Creating AperPhotResult for {reducedfit}, {astrosource}, {aperpix}, {pairs}.")
            result = cls(reducedfit=reducedfit, astrosource=astrosource, aperpix=aperpix, pairs=pairs)

        for key, value in kwargs.items():
            setattr(result, key, value)

        result.save()

