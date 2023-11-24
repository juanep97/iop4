import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

from django.db import models

from abc import ABCMeta, abstractmethod

import os
import io
import base64
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch, LogStretch, AsymmetricPercentileInterval
import astropy.io.fits as fits
from iop4lib.utils.filedproperty import FiledProperty, filed_property

import logging
logger = logging.getLogger(__name__)

class AbstractModelMeta(ABCMeta, type(models.Model)):
    pass

class AbstractModel(models.Model, metaclass=AbstractModelMeta):    
    class Meta:
        abstract = True


class FitFileModel(AbstractModel):
    """
    TODO: this should be an abstract class (abc) and model which can be inherited.
    This class adds some common functionality for fit files (get image preview, get histrogram, statistics, etc)
    """

    class Meta:
        abstract = True

    # Abstract properties

    # Identity

    @property
    @abstractmethod
    def epoch(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def filename(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def fileloc(self):
        raise NotImplementedError

    # File paths

    @property
    @abstractmethod
    def filepath(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def filedpropdir(self):
        raise NotImplementedError

    # Helper properties

    @property
    def fileexists(self):
        return os.path.isfile(self.filepath)

    @property
    def header(self):
        """
        Returns header.
        """        
        return fits.getheader(self.filepath)

    @property
    def data(self):
        """
        Returns data.
        """
        return fits.getdata(self.filepath)

    @property
    def mdata(self):
        """
        Return masked data (finite values).
        """
        data = fits.getdata(self.filepath)
        return np.ma.array(data, mask=~np.isfinite(data))

    @property
    def width(self):
        """
        Returns the width of the image.
        """
        return self.header['NAXIS1']
    
    @property
    def height(self):
        """
        Returns the height of the image.
        """
        return self.header['NAXIS2']

    # Common helper properties

    @property
    def stats(self):
        """
        Returns a dict with some important statistics about the data (mean, etc).
        """
        
        if not os.path.exists(self.filepath):
            return dict()
        
        from iop4lib.utils import stats_dict

        with fits.open(self.filepath) as hdul:
            data = hdul[0].data.flatten()

        data = data[np.isfinite(data)]

        return stats_dict(data)
        
    # Common helper methods for visualizing data in the web admin

    def get_imgbytes_preview_image(self, force_rebuild=False, **kwargs):
        """
        Build an image preview (png) of the data.

        Parameters
        ----------
        normtype : str
            'log', 'logstretch', or a matplotlib norm object.
        a : float
            If norm is 'logstretch', this is the stretch factor.
        vmin, vmax: float
            If norm is 'log', these are the min and max values for the norm.


        If called with default arguments (no kwargs) it will try to load from the disk,
        except if called with force_rebuild.

        When called with default arguments (no kwargs), if rebuilt, it will save the image to disk.
        """

        # logger.debug(f"Requested image preview for {self.fileloc}, {kwargs=}")
                     
        imgdata = self.mdata

        width = kwargs.get('width', 256)
        height = kwargs.get('height', 256)
        normtype = kwargs.get('norm', "log")
        vmin = kwargs.get('vmin', np.quantile(imgdata.compressed(), 0.3))
        vmax = kwargs.get('vmax', np.quantile(imgdata.compressed(), 0.99))
        a = kwargs.get('a', 10)

        fpath = os.path.join(self.filedpropdir, "img_preview_image.png")

        if len(kwargs) == 0 and not force_rebuild:
            if os.path.isfile(fpath) and os.path.getmtime(self.filepath) < os.path.getmtime(fpath):
                # logger.debug(f"Loading image preview for {self.fileloc} from disk")
                with open(fpath, 'rb') as f:
                    return f.read()

        # logger.debug(f"Building image preview for {self.fileloc}")

        cmap = plt.cm.gray.copy()
        cmap.set_bad(color='red')
        cmap.set_under(color='black')
        cmap.set_over(color='white')
            
        if normtype == "log":
            norm = ImageNormalize(imgdata, vmin=vmin, vmax=vmax, stretch=LogStretch(a=a))
        elif normtype == "logstretch":
            norm = ImageNormalize(stretch=LogStretch(a=a))


        buf = io.BytesIO()

        fig = mplt.figure.Figure(figsize=(width/100, height/100), dpi=iop4conf.mplt_default_dpi)
        ax = fig.subplots()
        ax.imshow(imgdata, cmap=cmap, origin='lower', norm=norm)
        try:
            from iop4lib.db import ReducedFit, AstroSource
            from iop4lib.enums import SRCTYPES
            # If it is a astro calibrated reduced fit, mark the src position
            if self.has_flag(ReducedFit.FLAGS.BUILT_REDUCED):
                if (target_src := self.sources_in_field.exclude(srctype=SRCTYPES.CALIBRATOR).get()) is not None:
                    target_pos_px = target_src.coord.to_pixel(self.wcs1)
                    ax.axhline(y=target_pos_px[1], color='r', linestyle="--", linewidth=1)
                    ax.axvline(x=target_pos_px[0], color='r', linestyle="--", linewidth=1)
        except Exception as e: 
            # it can fail if there is any problem with the fit calibration (e.g. in early versions the wcs
            # was not saved into key A, but we still want to be able to explore the data in the admin).
            logger.warning(f"Coudl not mark the target source position on ReducedFit {self.pk}: {e}")

        ax.axis('off')
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        fig.clf()

        # logger.debug(f"Image preview for {self.fileloc} built")

        buf.seek(0)
        imgbytes = buf.read()

        # if it was rebuilt, save it to disk if it is the default image settings.
        if len(kwargs) == 0:
            # logger.debug(f"Saving image preview for {self.fileloc} to disk")
            if not os.path.exists(self.filedpropdir):
                os.makedirs(self.filedpropdir)
            with open(fpath, 'wb') as f:
                f.write(imgbytes)

        return imgbytes
        
            
    def get_imgbytes_preview_histogram(self, force_rebuild=False, **kwargs):
        """
        If called with default arguments (no kwargs) it will try to load from the disk,
        except if called with force_rebuild.

        When called with default arguments (no kwargs), if rebuilt, it will save the image to disk.
        """

        # logger.debug(f"Requested preview histogram for {self.fileloc}, {kwargs=}")

        width = kwargs.get('width', 256)
        height = kwargs.get('height', 120)
        xscale = kwargs.get('xscale', "log")
        yscale = kwargs.get('yscale', "log")
        a = kwargs.get('a', 10)

        fpath = os.path.join(self.filedpropdir, "img_preview_histogram.png")

        if len(kwargs) == 0 and not force_rebuild:
            if os.path.isfile(fpath) and os.path.getmtime(self.filepath) < os.path.getmtime(fpath):
                #logger.debug(f"Loading preview histogram for {self.fileloc} from disk")
                with open(fpath, 'rb') as f:
                    return f.read()
         
        # logger.debug(f"Building preview histogram for {self.fileloc}")

        imgdata = self.mdata.flatten()

        buf = io.BytesIO()

        fig = mplt.figure.Figure(figsize=(width/100, height/100), dpi=iop4conf.mplt_default_dpi)
        ax = fig.subplots()

        ax.hist(imgdata, bins='sqrt', log=True, histtype='step')

        for q in np.quantile(imgdata, [0.5]):
            ax.axvline(x=q, color='r', linestyle="--", linewidth=1, label=f"{q:.2f}")

        # set axes    
        if xscale == "log":        
            vabsmin = np.amin(imgdata)
            vabsmax = np.amax(imgdata)
                
            a = 10

            def fscale(x):
                xs = (x - vabsmin) / (vabsmax - vabsmin)
                y = np.log(a * xs + 1) / np.log(a + 1)
                return y

            def fscale_inv(y):
                xs = (np.exp(y * np.log(a+1)) - 1) / a
                x = xs * (vabsmax - vabsmin) + vabsmin
                return x
            
            ax.set_xscale('function', functions=(fscale, fscale_inv))
            ax.set_xlim([vabsmin, vabsmax])  
        else:
            raise Exception("Not expecting xscale other than log")

        ax.set_yscale(yscale)

        # turn off axis
        ax.axis('off')

        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        fig.clf()

        # logger.debug(f"Preview histogram for {self.fileloc} built")

        buf.seek(0)
        imgbytes = buf.read()
        
        # if it was rebuilt, save it to disk if it is the default image settings.
        if len(kwargs) == 0:
            # logger.debug(f"Saving preview histogram for {self.fileloc} to disk")
            if not os.path.exists(self.filedpropdir):
                os.makedirs(self.filedpropdir)
            with open(fpath, 'wb') as f:
                f.write(imgbytes)
                    
        return imgbytes
        






""" # how it would be done without matplotlib
imgdata = LogNorm(vmin=vmin, vmax=vmax)(imgdata)

imgdata = 256*imgdata
imgdata = imgdata.astype(np.uint8)
image = Image.fromarray(imgdata).resize((width,height))


buf = io.BytesIO()
image.save(buf, format='png')
buf.seek(0)
imgbytes = buf.read()
"""