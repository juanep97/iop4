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
import astropy.units as u
import math
from photutils.profiles import RadialProfile, CurveOfGrowth

# iop4lib imports
from ..enums import PAIRS

# logging
import logging
logger = logging.getLogger(__name__)

from iop4lib.typing import *

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
    fwhm = models.FloatField(null=True, blank=True, help_text="FWHM of the source in arcseconds.")
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
        return f'<{self.__class__.__name__} {self.id} | {self.reducedfit.fileloc} {self.astrosource.name} ({self.aperpix:.1f}, {self.r_in:.1f}, {self.r_out:.1f}) px {self.pairs}>' 

    @classmethod
    def create(cls, reducedfit, astrosource, aperpix, r_in, r_out, pairs, re_use=True, **kwargs):

        if re_use: # TODO: conflicts with unique constraint
            result = AperPhotResult.objects.filter(
                reducedfit = reducedfit,
                astrosource = astrosource,
                aperpix = aperpix,
                r_in = r_in,
                r_out = r_out,
                pairs = pairs,
            ).first()
        else:
            result = None

        if result:
            logger.debug(f"AperPhotResult for {reducedfit}, {astrosource}, ({aperpix}, {r_in}, {r_out}), {pairs} already exists, it will be used instead.")
        else:
            logger.debug(f"Creating AperPhotResult for {reducedfit}, {astrosource}, ({aperpix}, {r_in}, {r_out}), {pairs}.")
            result = cls(reducedfit=reducedfit, astrosource=astrosource, aperpix=aperpix, r_in=r_in, r_out=r_out, pairs=pairs)

        for key, value in kwargs.items():
            setattr(result, key, value)

        result.save()

        return result

    @property
    def filedpropdir(self):
        return os.path.join(iop4conf.datadir, "aperphotresults", str(self.id))

    # plot helpers

    @property
    def default_plot_cutout(self):

        cutout_size = np.ceil(2.2*self.r_out)

        if self.reducedfit.has_pairs:
            cutout_size = max(cutout_size, 2.2*np.ceil(np.linalg.norm(self.reducedfit.hint_disp_sign_mean)))

        cutout = Cutout2D(
            self.reducedfit.mdata,
            position = (self.x_px, self.y_px),
            size = (cutout_size, cutout_size),
            wcs = (self.reducedfit.wcs1 if self.pairs == 'O' else self.reducedfit.wcs2),
        )

        return cutout

    def plot(self, **kwargs):

        fig = kwargs.get('fig') or plt.gcf()
        ax = kwargs.get('ax') or plt.gca()

        cutout = kwargs.get('cutout') or self.default_plot_cutout
        
        normtype = kwargs.get('norm', "log")
        vmin = kwargs.get('vmin', np.quantile(cutout.data.compressed(), 0.3))
        vmax = kwargs.get('vmax', np.quantile(cutout.data.compressed(), 0.99))
        a = kwargs.get('a', 10)

        if normtype == "log":
            norm = ImageNormalize(cutout.data.compressed(), vmin=vmin, vmax=vmax, stretch=LogStretch(a=a))
        elif normtype == "logstretch":
            norm = ImageNormalize(stretch=LogStretch(a=a))
        
        cmap = plt.cm.gray.copy()
        cmap.set_bad(color='red')
        cmap.set_under(color='black')
        cmap.set_over(color='white')

        # get wcs and centroid positions
        wcs_px_pos = self.astrosource.coord.to_pixel(cutout.wcs)
        xy_px_pos = cutout.to_cutout_position((self.x_px, self.y_px))

        # build the apertures
        ap = CircularAperture(xy_px_pos, r=self.aperpix)
        annulus = CircularAnnulus(xy_px_pos, r_in=self.r_in, r_out=self.r_out)
    
        # plot
        ax.imshow(cutout.data, cmap=cmap, origin='lower', norm=norm)
        ax.plot(wcs_px_pos[0], wcs_px_pos[1], 'rx', label='wcs')
        ax.plot(xy_px_pos[0], xy_px_pos[1], 'bo', label='centroid')
        ap.plot(ax, color='blue', lw=2, alpha=1)
        annulus.plot(ax, color='green', lw=2, alpha=1)

        # add 10'' length scale
        arcsecs = 10*u.arcsec
        pixs = (arcsecs / self.reducedfit.pixscale).to_value('pix')
        p1 = ax.transData.inverted().transform(ax.transAxes.transform((0.05, 0.01)))
        p2 = p1 + np.array([pixs, 0])
        ax.plot(
            (p1[0], p2[0]),
            (p1[1], p2[1]),
            color = 'r',
            linestyle = "-",
            linewidth = 2,
            alpha = 1,
        )
        ax.text(
            x=0.05, y=0.05,
            transform=ax.transAxes,
            s=f"{arcsecs.value:.0f}'' = {pixs:.1f} px",
            ha="left", va="bottom",
            color="red", fontsize=12,
        )

        if self.reducedfit.has_pairs:
            try:
                from iop4lib.db import PhotoPolResult
                other_pair = 'E' if self.pairs == 'O' else 'O'
                other_apf = PhotoPolResult.objects.get(
                    aperphotresults__in=[self.pk],
                ).aperphotresults.get(
                    reducedfit = self.reducedfit,
                    astrosource = self.astrosource,
                    aperpix = self.aperpix,
                    r_in = self.r_in,
                    r_out = self.r_out,
                    pairs = other_pair
                )
                other_c = cutout.to_cutout_position((other_apf.x_px, other_apf.y_px))
                other_mask = CircularAperture(other_c, r=self.aperpix).to_mask().to_image(cutout.shape).astype(bool)
                other_mask = np.ma.masked_where(~other_mask, other_mask)
                other_mask = np.ma.dstack([
                    other_mask,
                    np.zeros(other_mask.shape),
                    np.zeros(other_mask.shape),
                    np.full(other_mask.shape, 0.3),
                ])
                ax.imshow(other_mask, origin="lower")
            except Exception as e:
                pass

        return fig, ax

    def plot_rp(self, **kwargs):

        fig = kwargs.get('fig') or plt.gcf()
        ax = kwargs.get('ax') or plt.gca()

        cutout = kwargs.get('cutout') or self.default_plot_cutout

        c = cutout.to_cutout_position((self.x_px, self.y_px))

        fwhm = (self.fwhm*u.arcsec).to_value('pix', equivalencies=self.reducedfit.pixscale_equiv)
        sigma = fwhm / (2*math.sqrt(2*math.log(2)))

        rmax = min(cutout.shape)/2
        radii = np.arange(0, rmax)
        rp = RadialProfile(cutout.data, c, radii)

        fwhm_fit = rp.gaussian_fwhm

        cog = CurveOfGrowth(cutout.data, c, radii[1:])
        cog.normalize("max")

        ax.plot(rp.radius, rp.profile, 'k-', label="$I(r)$")

        ax.plot(rp.radius, rp.gaussian_fit(rp.radius), 'k:', label="Gaussian Fit")

        ax.axvline(x=fwhm/2, color='b', linestyle='-', linewidth=1, alpha=1, label="FWHM/2")
        ax.axvline(x=fwhm_fit/2, color='k', linestyle='--', linewidth=1, alpha=1, label="FWHM/2 (from fit)")
        ax.axvline(x=1*sigma, color='r', linestyle='-', linewidth=1, alpha=1, label="1$\sigma$")

        ax.axvline(x=3*sigma, color='r', linestyle='-', linewidth=1, alpha=1, label="3$\sigma$")
        ax.axvline(x=5*sigma, color='r', linestyle='-', linewidth=1, alpha=1, label="5$\sigma$")

        ax.set_xlabel("r [px]")

        ax.legend(loc="upper right")

        ax2 = ax.twinx()
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.plot(cog.radius, cog.profile, label="$F(r)$", color='tab:blue')
        ax2.legend(loc="lower right")
        
        return fig, ax

    def plot_proj(self, axis: Literal['x', 'y'], **kwargs):

        fig = kwargs.get('fig') or plt.gcf()
        ax = kwargs.get('ax') or plt.gca()

        cutout = kwargs.get('cutout') or self.default_plot_cutout

        c = cutout.to_cutout_position((self.x_px, self.y_px))

        # collapse image along X/Y axis
        if axis == 'x':
            i_ax = 0
            _line = ax.axvline
        elif axis == 'y':
            i_ax = 1
            _line = ax.axhline
        else:
            raise
        
        proj = np.nanmean(cutout.data, axis=i_ax)
        proj_i = np.arange(len(proj))

        if axis == 'x':
            proj_data = proj_i, proj
        elif axis == 'y':
            proj_data = proj, proj_i

        ax.plot(*proj_data, 'k-', lw=1.5, label=f"{axis} projection")

        # centroid position
        _line(c[i_ax], color='b', linestyle='-', lw=1, alpha=0.8, label='centroid')

        # aperture / annulus markers

        _line(c[i_ax] - self.aperpix, color='b', linestyle='--', lw=1, alpha=0.8)
        _line(c[i_ax] + self.aperpix, color='b', linestyle='--', lw=1, alpha=0.8, label='aperture')

        _line(c[i_ax] - self.r_in, color='g', linestyle=':', lw=1, alpha=0.8)
        _line(c[i_ax] + self.r_in, color='g', linestyle=':', lw=1, alpha=0.8)

        _line(c[i_ax] - self.r_out, color='g', linestyle='-.', lw=1, alpha=0.8)
        _line(c[i_ax] + self.r_out, color='g', linestyle='-.', lw=1, alpha=0.8, label='annulus')

        # wcs position
        wcs_px_pos = self.astrosource.coord.to_pixel(cutout.wcs)
        _line(wcs_px_pos[i_ax], color='r', linestyle='-', lw=1, alpha=0.8, label='wcs')

        if axis == 'x':
            ax.set_xlabel("x [pix]")
            ax.set_ylabel("mean counts")
        elif axis == 'y':
            ax.set_ylabel("y [pix]")
            ax.set_xlabel("mean counts")

        ax.legend()

        return fig, ax
    
    def plot_all(self, fig=None):
        
        cutout_size = np.ceil(2.2*self.r_out)

        if self.reducedfit.has_pairs:
            cutout_size = max(cutout_size, 2.2*np.ceil(np.linalg.norm(self.reducedfit.hint_disp_sign_mean)))
            
        cutout = Cutout2D(
            self.reducedfit.mdata,
            position = (self.x_px, self.y_px),
            size = (cutout_size, cutout_size),
            wcs = (self.reducedfit.wcs1 if self.pairs == 'O' else self.reducedfit.wcs2),
        )

        fig = fig or plt.gcf()

        gs = fig.add_gridspec(2, 2, hspace=0.01, wspace=0.01)

        ax_img   = fig.add_subplot(gs[0, 0])
        ax_projy = fig.add_subplot(gs[0, 1], sharey=ax_img)
        ax_projx = fig.add_subplot(gs[1, 0], sharex=ax_img)
        ax_rprof = fig.add_subplot(gs[1, 1])

        self.plot(cutout=cutout, fig=fig, ax=ax_img)
        self.plot_proj(axis='x', cutout=cutout, fig=fig, ax=ax_projx)
        self.plot_proj(axis='y', cutout=cutout, fig=fig, ax=ax_projy)
        self.plot_rp(cutout=cutout, fig=fig, ax=ax_rprof)

        ax_img.xaxis.tick_top()
        ax_img.xaxis.set_label_position("top")
        ax_img.set_xlabel("x [px]")
        ax_img.set_ylabel("y [px]")

        ax_projy.xaxis.tick_top()
        ax_projy.xaxis.set_label_position("top")
        ax_projy.yaxis.tick_right()
        ax_projy.yaxis.set_label_position("right")

        # shrink radial profile axis around around its center so there is some 
        # spacing betwwen this one and the rest
        pos = ax_rprof.get_position()
        cx = (pos.x0 + pos.x1) / 2
        cy = (pos.y0 + pos.y1) / 2
        scale = 0.85
        w = pos.width * scale
        h = pos.height * scale
        # actually, shift to right instead of keeping center
        x0 = pos.x0+0.9*(pos.width-w)
        y0 = cy-h/2
        ax_rprof.set_position([x0, y0, w, h])

        fig.suptitle(f"{self}")

        return fig, gs

    def get_img(self, force_rebuild=True, **kwargs):
        """Build a preview image (png) of the aperture over source.

        If called with default arguments (no kwargs) it will try to load from 
        the disk, except if called with force_rebuild.

        When called with default arguments (no kwargs), if rebuilt, it will save
        the image to disk.

        I will also show the wcs source position, the centroid (used as aperture
        position), and an angular scale.
        """
         
        fpath = os.path.join(self.filedpropdir, "img_preview_image.png")

        if len(kwargs) == 0 and not force_rebuild:
            if os.path.isfile(fpath) and os.path.getmtime(self.filepath) < os.path.getmtime(fpath):
                with open(fpath, 'rb') as f:
                    return f.read()
                
        # otherwise, build image again

        cutout = kwargs.get('cutout') or self.default_plot_cutout

        width = kwargs.get('width', 256)
        height = kwargs.get('height', 256)

        logger.info(f"Building image preview for AperPhotResult {self.id}.")

        buf = io.BytesIO()

        fig = mplt.figure.Figure(figsize=(width/100, height/100), dpi=iop4conf.mplt_default_dpi)
        ax = fig.subplots()

        self.plot(cutout=cutout, fig=fig, ax=ax)

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

    def get_img_all(self, **kwargs):

        width = kwargs.get('width', 1024)
        height = kwargs.get('height', 1024)

        buf = io.BytesIO()

        fig = mplt.figure.Figure(figsize=(width/100, height/100), dpi=iop4conf.mplt_default_dpi)
        self.plot_all(fig=fig)
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        fig.clf()

        buf.seek(0)
        imgbytes = buf.read()

        return imgbytes
