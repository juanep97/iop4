# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
from abc import ABCMeta, abstractmethod

import re
import ftplib
import logging
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
import astrometry

# iop4lib imports
from iop4lib.enums import *

# logging
import logging
logger = logging.getLogger(__name__)

class Telescope(metaclass=ABCMeta):
    """
        Inherit this class to provide telescope specific functionality and
        translations.

        Attributes and methods that must be implemented are marked as abstract (they will give
        error if the class is inherited and the method is not implemented in the subclass).

        Some classmethods are already defined since they are telescope independent; we should not need to
        override them (but it can be done).

        .. note::
            This class is abstract, it should not be instantiated.

        .. note::
            To add a new telescope, inherit this class and add it to the list of known telescopes
            in the get_known() method and the TelescopeEnum.
    """

    # Abstract attributes

    # telescope identification

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def abbrv(self):
        pass

    @property
    @abstractmethod
    def telescop_kw(self):
        pass

    # telescope properties

    @property
    @abstractmethod
    def gain_e_adu(self):
        pass

    # Abstract methods

    @classmethod
    @abstractmethod
    def list_remote_raw_fnames(cls, epoch):
        pass

    @classmethod
    @abstractmethod
    def download_rawfits(cls, epoch):
        pass

    @classmethod
    @abstractmethod
    def list_remote_epochnames(cls):
        pass

    @classmethod
    @abstractmethod
    def classify_juliandate_rawfit(cls, rawfit):
        pass

    @classmethod
    @abstractmethod
    def classify_imgtype_rawfit(cls, rawfit):
        pass

    @classmethod
    @abstractmethod
    def classify_band_rawfit(cls, rawfit):
        pass

    @classmethod
    @abstractmethod
    def classify_obsmode_rawfit(cls, rawfit):
        pass

    @classmethod
    @abstractmethod
    def get_header_hintcoord(cls, rawfit, *args, **kwargs):
        pass
    
    @classmethod
    @abstractmethod
    def get_astrometry_position_hint(cls, rawfit, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def get_astrometry_size_hint(cls, rawfit):
        pass

    # Class methods (usable)
    
    @classmethod
    def get_known(cls):
        from .cahat220 import CAHAT220
        from .osnt090 import OSNT090
        return [OSNT090, CAHAT220]
    
    @classmethod
    def by_name(cls, name):
        """
        Try to get telescope by name, then abbreviation, else raise Exception.
        """
        for tel in Telescope.get_known():
            if tel.name == name:
                return tel
        for tel in Telescope.get_known():
            if tel.abbrv == name:
                return tel
        raise NotImplementedError(f"Telescope {name} not implemented.")
        
    @classmethod
    def is_known(self, name):
        """
        Check if a telescope is known by name or abbreviation.
        """
        return (name in [tel.name for tel in Telescope.get_known()]) or (name in [tel.abbrv for tel in Telescope.get_known()])
    

    # telescope independent functionality (they don't need to be overriden in subclasses)


    @classmethod
    def check_telescop_kw(cls, rawfit):
        """
        TELESCOP is an standard FITS keyword, it should not be telescope dependent.
        """
        telescop_header = fits.getheader(rawfit.filepath, ext=0)["TELESCOP"] 

        if (telescop_header != cls.telescop_kw):
            raise ValueError(f"TELESCOP in fits header ({telescop_header}) does not match telescope class kw ({cls.telescop_kw}).")
        
    @classmethod
    def classify_instrument_kw(cls, rawfit):
        """
        INSTRUME is an standard FITS keyword, it should not be telescope dependent.
        """

        instrume_header = fits.getheader(rawfit.filepath, ext=0)["INSTRUME"] 
        
        if instrume_header == "AndorT90":
            rawfit.instrument = INSTRUMENTS.AndorT90
        elif instrume_header == "CAFOS 2.2":
            rawfit.instrument = INSTRUMENTS.CAFOS
        else:
            raise ValueError(f"INSTRUME in fits header ({instrume_header}) not known.")
    
    @classmethod
    def classify_imgsize(cls, rawfit):
        import astropy.io.fits as fits
        from iop4lib.db import RawFit

        with fits.open(rawfit.filepath) as hdul:
            if hdul[0].header["NAXIS"] == 2:
                sizeX = hdul[0].header["NAXIS1"]
                sizeY = hdul[0].header["NAXIS2"]
                rawfit.imgsize = f"{sizeX}x{sizeY}"
                return rawfit.imgsize
            else:
                raise ValueError(f"Raw fit file {rawfit.fileloc} has NAXIS != 2, cannot get imgsize.")
            
    @classmethod
    def classify_exptime(cls, rawfit):
        """
        EXPTIME is an standard FITS keyword, measured in seconds.
        """
        import astropy.io.fits as fits
        from iop4lib.db import RawFit

        with fits.open(rawfit.filepath) as hdul:
            rawfit.exptime = hdul[0].header["EXPTIME"]


    @classmethod
    def get_header_objecthint(self, rawfit):
        """ Get a hint for the AstroSource in this image from the header. OBJECT is a standard keyword. Return None if none found. 
        
        At the moment his only tries to match sources
        with the IAU name format [0-9]*\+[0-9]*.
        """
        
        from iop4lib.db import AstroSource

        object_header = rawfit.header["OBJECT"]
        
        matchs = re.findall(r".*?([0-9]*\+[0-9]*).*", object_header)
        if len(matchs) > 0:
            return AstroSource.objects.filter(name__contains=matchs[0]).first()
        else:
            return None


