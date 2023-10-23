# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
from abc import ABCMeta, abstractmethod
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

# iop4lib imports
from iop4lib.enums import *
from .telescope import Telescope, FTPArchiveMixin

# logging
import logging
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from iop4lib.db import RawFit, ReducedFit, Epoch

class CAHAT220(FTPArchiveMixin, Telescope, metaclass=ABCMeta):
    """
    CAHA T220 telescope.

    CAHA has a per-program ftp login (PI user and password in config).
    The file structure when you login is a list of files as:
    {yymmdd}_CAFOS/
    where CAFOS refers to the polarimeter
    and inside each folder there are the files for that day.

    Currently only one instrument, CAFOS.
    """

    # telescope identification

    name = "CAHA-T220"
    abbrv = "T220"
    telescop_kw = "CA-2.2"

    # telescope specific properties

    ftp_address = iop4conf.caha_address
    ftp_user = iop4conf.caha_user
    ftp_password = iop4conf.caha_password
    ftp_encoding = 'utf-8'

    re_expr_dirnames = re.compile(r"([0-9]{2}[0-9]{2}[0-9]{2})_CAFOS", flags=re.IGNORECASE)
    re_expr_fnames = re.compile(r".*\.fi?ts?", flags=re.IGNORECASE)

    # telescope specific methods

    # for the ftp archive mixin 

    @classmethod
    def remote_dirname_to_epochname(cls, dirname):
        matchs = cls.re_expr_dirnames.findall(dirname)
        if len(matchs) != 1:
            return None
        else:
            return f"{cls.name}/20{matchs[0][0:2]}-{matchs[0][2:4]}-{matchs[0][4:6]}"
        
    @classmethod
    def epochname_to_remote_dirname(cls, epochname):
        from iop4lib.db import Epoch
        tel, night = Epoch.epochname_to_tel_night(epochname)
        return night.strftime("%y%m%d_CAFOS")