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

    ftp_address = iop4conf.osn_t150_address
    ftp_user = iop4conf.osn_t150_user
    ftp_password = iop4conf.osn_t150_password

    # telescope specific methods

    @classmethod
    def get_astrometry_size_hint(cls, rawfit):
        r""" Get the size hint for this telescope / rawfit.

            According to OSN T0150 camera information (https://www.osn.iaa.csic.es/page/camaras-ccdt150-y-ccdt90) 
            camera pixels are 0.232as/px and it has a field of view of 7.92' x 7.92'.
            If the files are 1x1 it will be that, if they are 2x2 it will be twice.
        """

        if rawfit.header['NAXIS1'] == 2048:
            return astrometry.SizeHint(lower_arcsec_per_pixel=0.95*cls.andort150_arcsec_per_pix, upper_arcsec_per_pixel=1.05*cls.andort150_arcsec_per_pix)
        elif rawfit.header['NAXIS1'] == 1024:
            return astrometry.SizeHint(lower_arcsec_per_pixel=2*0.95*cls.andort150_arcsec_per_pix, upper_arcsec_per_pixel=2*1.05*cls.andort150_arcsec_per_pix)
        

    @classmethod
    def compute_relative_photometry(cls, rawfit):
        logger.warning(f"OSNT150.compute_relative_photometry not implemented yet, using OSNT090.compute_relative_photometry {super(cls)=}")
        super(OSNT150, cls).compute_relative_photometry(rawfit)

    @classmethod
    def compute_relative_polarimetry(cls, polarimetry_group):
        logger.warning(f"OSNT150.compute_relative_polarimetry not implemented yet, using OSNT090.compute_relative_polarimetry {super(cls)=}")
        super(OSNT150, cls).compute_relative_polarimetry(polarimetry_group)

