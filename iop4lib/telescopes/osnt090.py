# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from abc import ABCMeta, abstractmethod

# other imports
import os
import re
import ftplib
from pathlib import Path
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
import astrometry
import numpy as np
import math
import datetime

# iop4lib imports
from iop4lib.enums import *
from .telescope import Telescope, FTPArchiveMixin

# logging
import logging
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from iop4lib.db import RawFit, ReducedFit, Epoch


class OSNT090(FTPArchiveMixin, Telescope, metaclass=ABCMeta):
    
    # telescope identification

    name = "OSN-T090"
    abbrv = "T090"
    telescop_kw = "T90-OSN"

    # telescope specific properties

    ftp_address = iop4conf.osn_t090_address
    ftp_user = iop4conf.osn_t090_user
    ftp_password = iop4conf.osn_t090_password
    ftp_encoding = 'latin-1'

    re_expr_dirnames = re.compile(r"([0-9]{4}[0-9]{2}[0-9]{2})", flags=re.IGNORECASE)
    re_expr_fnames = re.compile('|'.join(iop4conf.osn_fnames_patterns), flags=re.IGNORECASE)

    # telescope specific methods

    @classmethod
    def check_telescop_kw(cls, rawfit):
        r""" Subclassed to account for DIPOL files, that may have empty TELESCOP keyword as of 2023-10-11 
        
        If it is empty, check first the instrument, and if it is DIPOL and the night is before 2023-10-11, then continue.

        Otherwise just call the parent method.
        """
        if rawfit.header["TELESCOP"] == "" and rawfit.night < datetime.date(2023, 10, 11):
            cls.classify_instrument_kw(rawfit)
            if rawfit.instrument == INSTRUMENTS.DIPOL:
                return
            
        super().check_telescop_kw(rawfit)

    # for the ftp archive mixin

    @classmethod 
    def remote_dirname_to_epochname(cls, dirname):
        matchs = cls.re_expr_dirnames.findall(dirname)
        if len(matchs) != 1:
            return None
        else:
            return f"{cls.name}/{matchs[0][0:4]}-{matchs[0][4:6]}-{matchs[0][6:8]}"

    @classmethod
    def epochname_to_remote_dirname(cls, epochname):
        from iop4lib.db import Epoch
        tel, night = Epoch.epochname_to_tel_night(epochname)
        return night.strftime("%Y%m%d")