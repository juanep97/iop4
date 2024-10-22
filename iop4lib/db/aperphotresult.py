# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

 # django imports
from django.db import models

# other imports
import os
import io
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch
from astropy.nddata import Cutout2D
from photutils.aperture import CircularAperture, CircularAnnulus

# iop4lib imports
from ..enums import *
from iop4lib.instruments.instrument import Instrument

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
    x_px = models.FloatField(null=True, help_text="used pixel position of the source in the image, x coordinate.")
    y_px = models.FloatField(null=True, help_text="used pixel position of the source in the image, y coordinate.")
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

        return result

    @property
    def filedpropdir(self):
        return os.path.join(iop4conf.datadir, "aperphotresults", str(self.id))
    
    def get_img(self, force_rebuild=True, **kwargs):
        """
        Build an image preview (png) of the aperture and annulus over the source.

        If called with default arguments (no kwargs) it will try to load from the disk,
        except if called with force_rebuild.

        When called with default arguments (no kwargs), if rebuilt, it will save the image to disk.
        """
                     
        wcs = self.reducedfit.wcs1 if self.pairs == 'O' else self.reducedfit.wcs2

        if self.reducedfit.has_pairs:
            cutout_size = np.ceil(2.2*np.linalg.norm(Instrument.by_name(self.reducedfit.instrument).disp_sign_mean))
        else:
            cutout_size = np.ceil(1.3*self.r_out)

        cutout = Cutout2D(self.reducedfit.mdata, (self.x_px, self.y_px), (cutout_size, cutout_size), wcs)

        width = kwargs.get('width', 256)
        height = kwargs.get('height', 256)
        normtype = kwargs.get('norm', "log")
        vmin = kwargs.get('vmin', np.quantile(cutout.data.compressed(), 0.3))
        vmax = kwargs.get('vmax', np.quantile(cutout.data.compressed(), 0.99))
        a = kwargs.get('a', 10)

        fpath = os.path.join(self.filedpropdir, "img_preview_image.png")

        if len(kwargs) == 0 and not force_rebuild:
            if os.path.isfile(fpath) and os.path.getmtime(self.filepath) < os.path.getmtime(fpath):
                with open(fpath, 'rb') as f:
                    return f.read()

        cmap = plt.cm.gray.copy()
        cmap.set_bad(color='red')
        cmap.set_under(color='black')
        cmap.set_over(color='white')
            
        if normtype == "log":
            norm = ImageNormalize(cutout.data.compressed(), vmin=vmin, vmax=vmax, stretch=LogStretch(a=a))
        elif normtype == "logstretch":
            norm = ImageNormalize(stretch=LogStretch(a=a))


        buf = io.BytesIO()

        fig = mplt.figure.Figure(figsize=(width/100, height/100), dpi=iop4conf.mplt_default_dpi)
        ax = fig.subplots()

        wcs_px_pos = self.astrosource.coord.to_pixel(cutout.wcs)
        xy_px_pos = cutout.to_cutout_position((self.x_px, self.y_px))
        ap = CircularAperture(xy_px_pos, r=self.aperpix)
        annulus = CircularAnnulus(xy_px_pos, r_in=self.r_in, r_out=self.r_out)
    
        ax.imshow(cutout.data, cmap=cmap, origin='lower', norm=norm)
        ax.plot(wcs_px_pos[0], wcs_px_pos[1], 'rx', label='WCS')
        ax.plot(xy_px_pos[0], xy_px_pos[1], 'bo', label='Photometry')
        ap.plot(ax, color='blue', lw=2, alpha=1)
        annulus.plot(ax, color='green', lw=2, alpha=1)

        ax.axis('off')
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        fig.clf()

        buf.seek(0)
        imgbytes = buf.read()

        # if it was rebuilt, save it to disk if it is the default image settings.
        if len(kwargs) == 0:
            if not os.path.exists(self.filedpropdir):
                os.makedirs(self.filedpropdir)
            with open(fpath, 'wb') as f:
                f.write(imgbytes)

        return imgbytes