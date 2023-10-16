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
import numpy as np
import math

# iop4lib imports
from iop4lib.enums import *

# logging
import logging
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from iop4lib.db import RawFit, ReducedFit, Epoch    

class Telescope(metaclass=ABCMeta):
    """ Base class for telescopes.

        Inherit this class to provide telescope specific functionality (e.g. discovering and 
        downloading new data, classification of instruments, etc).

        Attributes and methods that must be implemented are marked as abstract (they will give
        error if the class is inherited and the method is not implemented in the subclass).

        Other methods can be implemented in the subclass but are not required just raise NotImplementedError.

        Some classmethods are already defined since they are telescope independent; we should not need to
        override them (but it can be done).

        .. note::
            This class is abstract, it should not be instantiated.

        .. note::
            To add a new telescope, inherit this class and add it to the list of known telescopes
            in the get_known() method and the TelescopeEnum.
    """

    # Abstract attributes

    # these attributes must be implemented in the subclass

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


    # Abstract methods

    # This methods must be implemented in the subclass

    @classmethod
    @abstractmethod
    def list_remote_raw_fnames(cls, epoch: 'Epoch') -> list[str] :
        pass

    @classmethod
    @abstractmethod
    def download_rawfits(cls, epoch: 'Epoch') -> None :
        pass

    @classmethod
    @abstractmethod
    def list_remote_epochnames(cls) -> list[str] :
        pass

    @classmethod
    @abstractmethod
    def list_remote_filelocs(cls, epochnames: list[str]) -> list[str] :
        pass

    # Class methods (you should be using these only from this Telescope class, not from subclasses)
    
    @classmethod
    def get_known(cls):
        from .cahat220 import CAHAT220
        from .osnt090 import OSNT090
        from .osnt150 import OSNT150
        return [CAHAT220, OSNT090, OSNT150]
    
    @classmethod
    def by_name(cls, name: str) -> 'Telescope':
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
    
    # telescope independent functionality
    # you should be using these from the subclasses already
    # these don't need to be overriden in subclasses, but they can be (e.g. OSN-T090 overrides check_telescop_kw)

    @classmethod
    def classify_rawfit(cls, rawfit: 'RawFit'):
        r""" Try to classify a RawFit object.

        This method will first check that the rawfit belongs to this telescope, 
        classify the instrument, then hand off classification to the instrument 
        class.        
        """

        from iop4lib.instruments import Instrument

        cls.check_telescop_kw(rawfit)
        cls.classify_instrument_kw(rawfit)
        Instrument.by_name(rawfit.instrument).classify_rawfit(rawfit)

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
        elif instrume_header == "Andor":
            rawfit.instrument = INSTRUMENTS.AndorT150
        elif instrume_header == "CAFOS 2.2":
            rawfit.instrument = INSTRUMENTS.CAFOS
        elif instrume_header == "ASI Camera (1)":
            rawfit.instrument = INSTRUMENTS.DIPOL
        else:
            raise ValueError(f"INSTRUME in fits header ({instrume_header}) not known.")
    