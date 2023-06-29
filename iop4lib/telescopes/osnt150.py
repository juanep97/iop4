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
from .telescope import Telescope
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

    arcsec_per_pix = 0.232
    gain_e_adu = 4.5
    field_width_arcmin = 7.92

    ftp_address = iop4conf.osn_t150_address
    ftp_user = iop4conf.osn_t150_user
    ftp_password = iop4conf.osn_t150_password

    # telescope specific methods

    @classmethod
    def compute_relative_photometry(cls, rawfit):
        logger.warning("OSNT150.compute_relative_photometry not implemented yet, using OSNT090.compute_relative_photometry")
        super(cls).compute_relative_photometry(rawfit)

    @classmethod
    def compute_relative_polarimetry(cls, polarimetry_group):
        logger.warning("OSNT150.compute_relative_polarimetry not implemented yet, using OSNT090.compute_relative_polarimetry")
        super(cls).compute_relative_polarimetry(polarimetry_group)

