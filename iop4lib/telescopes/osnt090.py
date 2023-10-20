# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from abc import ABCMeta, abstractmethod

# other imports
import os
import re
import ftplib
import logging
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
import astrometry
import numpy as np
import math
import datetime

# iop4lib imports
from iop4lib.enums import *
from .telescope import Telescope

# logging
import logging
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from iop4lib.db import RawFit, ReducedFit, Epoch


class OSNT090(Telescope, metaclass=ABCMeta):
    
    # telescope identification

    name = "OSN-T090"
    abbrv = "T090"
    telescop_kw = "T90-OSN"

    # telescope specific properties

    ftp_address = iop4conf.osn_t090_address
    ftp_user = iop4conf.osn_t090_user
    ftp_password = iop4conf.osn_t090_password

    fnames_re_expr = re.compile('|'.join(iop4conf.osn_fnames_patterns), flags=re.IGNORECASE)

    # telescope specific methods

    @classmethod
    def list_remote_epochnames(cls):
        try:
            logger.debug(f"Loging to {cls.name} FTP server")

            ftp =  ftplib.FTP(cls.ftp_address, encoding='latin-1')
            ftp.login(cls.ftp_user, cls.ftp_password)
            remote_dirnameL_all = ftp.nlst()
            ftp.quit()

            logger.debug(f"Total of {len(remote_dirnameL_all)} dirs in {cls.name} remote.")

            remote_epochnameL_all = list()

            for dirname in remote_dirnameL_all:
                matchs = re.findall(r"([0-9]{4}[0-9]{2}[0-9]{2})", dirname)
                if len(matchs) != 1:
                    logger.warning(f"Could not parse {dirname} as a valid epochname.")
                else:
                    remote_epochnameL_all.append(f"{cls.name}/{matchs[0][0:4]}-{matchs[0][4:6]}-{matchs[0][6:8]}")

            logger.debug(f"Filtered to {len(remote_epochnameL_all)} epochs in OSN.")

            return remote_epochnameL_all
        
        except Exception as e:
            raise Exception(f"Error listing remote epochs in {cls.name}: {e}.")


    @classmethod
    def list_remote_raw_fnames(cls, epoch):
        try:
            logger.debug(f"Loging to {cls.name} FTP server")

            ftp =  ftplib.FTP(cls.ftp_address, encoding='latin-1')
            ftp.login(cls.ftp_user, cls.ftp_password)

            logger.debug(f"Changing to OSN dir {epoch.yyyymmdd}")

            ftp.cwd(f"{epoch.yyyymmdd}")
            remote_fnameL_all = ftp.nlst()
            ftp.quit()

            logger.debug(f"Total of {len(remote_fnameL_all)} files in OSN {epoch.epochname}: {remote_fnameL_all}.")

            remote_fnameL = [s for s in remote_fnameL_all if cls.fnames_re_expr.search(s)] # Filter by filename pattern (get only our files)

            logger.debug(f"Filtered to {len(remote_fnameL)} files in OSN {epoch.epochname}.")

            return remote_fnameL
        
        except Exception as e:
            raise Exception(f"Error listing remote dir for {epoch.epochname}: {e}.")
        
    @classmethod
    def download_rawfits(cls, rawfits):
        try:
            ftp =  ftplib.FTP(cls.ftp_address)
            ftp.login(cls.ftp_user, cls.ftp_password)

            for rawfit in rawfits:
                logger.debug(f"Changing to OSN dir {rawfit.epoch.night} ...")
                ftp.cwd(f"{rawfit.epoch.yyyymmdd}")

                logger.debug(f"Downloading {rawfit.fileloc} from OSN archive ...")

                with open(rawfit.filepath, 'wb') as f:
                    ftp.retrbinary('RETR ' + rawfit.filename, f.write)

                ftp.cwd("..")
            
            ftp.quit()

        except Exception as e:
            raise Exception(f"Error downloading file {rawfit.filename}: {e}.")

    @classmethod
    def list_remote_filelocs(cls, epochnames: list[str]) -> list[str]:

        from iop4lib.db import Epoch

        ftp =  ftplib.FTP(cls.ftp_address, cls.ftp_user, cls.ftp_password, encoding='latin-1')

        dirnames = ftp.nlst()

        fileloc_list = list()
        
        for epochname in epochnames:

            tel, night = Epoch.epochname_to_tel_night(epochname)
            yyyymmdd = night.strftime("%Y%m%d")

            if yyyymmdd not in dirnames:
                logger.warning(f"Could not find {yyyymmdd} in {cls.name} remote.")
                continue

            try:
                
                fileloc_list.extend([f"{epochname}/{fname}" for fname in ftp.nlst(yyyymmdd) if cls.fnames_re_expr.search(fname) and fname != '.' and fname != '..'])
                            
            except Exception as e:
                logger.error(f"Error listing OSN remote dir for {epochname}: {e}.")

        ftp.quit()

        return fileloc_list

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
