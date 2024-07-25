# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from abc import ABCMeta, abstractmethod

# other imports
import re
import ftplib
import logging
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
import astrometry

# iop4lib imports
from iop4lib.enums import *
from .telescope import Telescope, ReadOnlyClassProperty
from .osnt090 import OSNT090

# logging
import logging
logger = logging.getLogger(__name__)

class OSNT150(OSNT090, Telescope, metaclass=ABCMeta):
    """OSN-T150 telescope class.
    
    Inherits from OSNT090 since the remote archive access, reduction processes, etc are almost identical.
    """
    
    # telescope identification

    name = "OSN-T150"
    abbrv = "T150"
    telescop_kw = "T150-OSN"

    # telescope specific properties

    # ftp connection details need to be properties so they can be overriden
    # otherwise they are defined at import time
    
    @ReadOnlyClassProperty
    def ftp_address(cls):
        return iop4conf.osn_t150_address
    
    @ReadOnlyClassProperty
    def ftp_user(cls):
        return iop4conf.osn_t150_user
    
    @ReadOnlyClassProperty
    def ftp_password(cls):
        return iop4conf.osn_t150_password
    
    @ReadOnlyClassProperty
    def ftp_encoding(cls):
        return 'utf-8'
