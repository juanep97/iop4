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
from .telescope import Telescope

# logging
import logging
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from iop4lib.db import RawFit, ReducedFit, Epoch

class CAHAT220(Telescope, metaclass=ABCMeta):
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

    fnames_re_expr = re.compile(r".*\.fi?ts?", flags=re.IGNORECASE)

    # telescope specific methods

    @classmethod
    def list_remote_epochnames(cls):
        try:
            logger.debug("Loging to CAHA FTP server.")

            ftp =  ftplib.FTP(iop4conf.caha_address)
            ftp.login(iop4conf.caha_user, iop4conf.caha_password)

            remote_dirnameL_all = ftp.nlst()
            ftp.quit()

            if '.' in remote_dirnameL_all:
                remote_dirnameL_all.remove('.')
            if '..' in remote_dirnameL_all:
                remote_dirnameL_all.remove('..')

            remote_epochnameL_all = list()

            for dirname in remote_dirnameL_all:
                matchs = re.findall(r"([0-9]{2}[0-9]{2}[0-9]{2})_CAFOS", dirname)
                if len(matchs) != 1:
                    logger.warning(f"Could not parse {dirname} as a valid epochname.")
                else:
                    remote_epochnameL_all.append(f"{cls.name}/20{matchs[0][0:2]}-{matchs[0][2:4]}-{matchs[0][4:6]}")

            logger.debug(f"Total of {len(remote_epochnameL_all)} epochs in CAHA.")

            return remote_epochnameL_all
        
        except Exception as e:
            raise Exception(f"Error listing remote epochs for {Telescope.name}: {e}.")

    @classmethod
    def list_remote_raw_fnames(cls, epoch):
        try:

            logger.debug("Loging to CAHA FTP server.")

            ftp =  ftplib.FTP(iop4conf.caha_address)
            ftp.login(iop4conf.caha_user, iop4conf.caha_password)

            try:
                logger.debug(f"Trying to change {epoch.yymmdd}_CAFOS/ directory.")
                ftp.cwd(f"{epoch.yymmdd}_CAFOS")
            except Exception as e:
                logger.debug(f"Could not change to {epoch.yymmdd}_CAFOS/ trying cd to {epoch.yymmdd}/ instead.")
                ftp.cwd(f"{epoch.yymmdd}")

            remote_fnameL_all = ftp.nlst()
            ftp.quit()

            if '.' in remote_fnameL_all:
                remote_fnameL_all.remove('.')
            if '..' in remote_fnameL_all:
                remote_fnameL_all.remove('..')

            logger.debug(f"Total of {len(remote_fnameL_all)} files in CAHA {epoch.epochname}.")

            remote_fnameL = [s for s in remote_fnameL_all if cls.fnames_re_expr.search(s)]
            
            logger.debug(f"Filtered to {len(remote_fnameL)} *.fit(s) files in CAHA {epoch.epochname}.")

            # all files are ours
            return remote_fnameL
        
        except Exception as e:
            raise Exception(f"Error listing remote dir for {epoch.epochname}: {e}.")
        
    @classmethod
    def download_rawfits(cls, rawfits):
        try:
            ftp =  ftplib.FTP(iop4conf.caha_address)
            ftp.login(iop4conf.caha_user, iop4conf.caha_password)

            for rawfit in rawfits:
                #logger.debug(f"Changing to CAHA dir {rawfit.epoch.yymmdd}_CAFOS ...")

                ftp.cwd(f"{rawfit.epoch.yymmdd}_CAFOS")

                logger.debug(f"Downloading {rawfit.fileloc} from CAHA archive ...")

                with open(rawfit.filepath, 'wb') as f:
                    ftp.retrbinary('RETR ' + rawfit.filename, f.write)

                ftp.cwd("..")

            ftp.quit()
        except Exception as e:
            raise Exception(f"Error downloading {rawfits}: {e}.")

    @classmethod
    def list_remote_filelocs(cls, epochnames: list[str]) -> list[str]:
        from iop4lib.db import Epoch

        ftp =  ftplib.FTP(iop4conf.caha_address)
        
        ftp.login(iop4conf.caha_user, iop4conf.caha_password)

        dirnames = ftp.nlst()

        fileloc_list = list()

        for epochname in epochnames:

            tel, night = Epoch.epochname_to_tel_night(epochname)
            yymmdd = night.strftime("%y%m%d")

            if f"{yymmdd}_CAFOS" not in dirnames:
                logger.error(f"CAHA remote dir {yymmdd}_CAFOS does not exist.")
                continue

            try:
                ftp.cwd(f"/{yymmdd}_CAFOS")

                fileloc_list.extend([f"{epochname}/{fname}" for fname in ftp.nlst() if cls.fnames_re_expr.search(fname) and fname != '.' and fname != '..'])
        

            except Exception as e:
                logger.error(f"Error listing CAHA remote dir for {epochname}: {e}.")
            
        ftp.quit()

        return fileloc_list