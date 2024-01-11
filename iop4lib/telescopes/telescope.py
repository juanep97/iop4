# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
from abc import ABCMeta, abstractmethod

import re
import os
from pathlib import Path
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
    def download_rawfits(cls, rawfits: list['RawFit']) -> None :
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
        
        if instrume_header == "RoperT90" and rawfit.epoch.night < datetime.date(2021, 10, 23):
            # RoperT90 was replaced by AndorT90 on 2021-10-23, but the control PC was not updated until some time later
            rawfit.instrument = INSTRUMENTS.RoperT90
        elif instrume_header == "AndorT90" or (instrume_header == "RoperT90" and rawfit.epoch.night >= datetime.date(2021, 10, 23)):
            rawfit.instrument = INSTRUMENTS.AndorT90
        elif instrume_header == "Andor" or instrume_header == "AndorT150": # until 2023-01-11, AndorT150 was called simply Andor
            rawfit.instrument = INSTRUMENTS.AndorT150
        elif instrume_header == "CAFOS 2.2":
            rawfit.instrument = INSTRUMENTS.CAFOS
        elif instrume_header == "ASI Camera (1)":
            rawfit.instrument = INSTRUMENTS.DIPOL
        else:
            raise ValueError(f"INSTRUME in fits header ({instrume_header}) not known.")
    







class FTPArchiveMixin():

    @classmethod
    @abstractmethod
    def remote_dirname_to_epochname(cls, dirname):
        pass
        
    @classmethod
    @abstractmethod
    def epochname_to_remote_dirname(cls, epochname):
        pass


    @classmethod
    def list_remote_epochnames(cls):
        try:
            logger.debug(f"Loging to {cls.name} FTP server.")

            ftp =  ftplib.FTP(cls.ftp_address, encoding=cls.ftp_encoding)
            ftp.login(cls.ftp_user, cls.ftp_password)

            remote_dirnameL_all = ftp.nlst()
            ftp.quit()

            if '.' in remote_dirnameL_all:
                remote_dirnameL_all.remove('.')
            if '..' in remote_dirnameL_all:
                remote_dirnameL_all.remove('..')

            logger.debug(f"Total of {len(remote_dirnameL_all)} dirs in {cls.name} remote.")

            remote_epochnameL_all = list()

            for dirname in remote_dirnameL_all:
                if (epochname := cls.remote_dirname_to_epochname(dirname)) is None:
                    logger.warning(f"Could not parse {dirname} as a valid epochname.")
                else:
                    remote_epochnameL_all.append(epochname)

            logger.debug(f"Filtered to {len(remote_epochnameL_all)} epochs in {cls.name}.")

            return remote_epochnameL_all
        
        except Exception as e:
            raise Exception(f"Error listing remote epochs for {Telescope.name}: {e}.")

    @classmethod
    def list_remote_raw_fnames(cls, epoch):
        try:

            logger.debug(f"Loging to {cls.name} FTP server.")

            ftp =  ftplib.FTP(cls.ftp_address, encoding=cls.ftp_encoding)
            ftp.login(cls.ftp_user, cls.ftp_password)

            remote_dir = cls.epochname_to_remote_dirname(epoch.epochname)
            logger.debug(f"Trying to change {remote_dir} directory.")
            ftp.cwd(f"/{remote_dir}")

            remote_fnameL_all = ftp.nlst()
            ftp.quit()

            if '.' in remote_fnameL_all:
                remote_fnameL_all.remove('.')
            if '..' in remote_fnameL_all:
                remote_fnameL_all.remove('..')

            logger.debug(f"Total of {len(remote_fnameL_all)} files in {cls.name} {epoch.epochname}: {remote_fnameL_all}.")

            remote_fnameL = [s for s in remote_fnameL_all if cls.re_expr_fnames.search(s)]
            
            logger.debug(f"Filtered to {len(remote_fnameL)} files in {cls.name} {epoch.epochname}.")

            return remote_fnameL
        
        except Exception as e:
            raise Exception(f"Error listing remote dir for {epoch.epochname}: {e}.")
        
    @classmethod
    def download_rawfits(cls, rawfits: list['RawFit'] | list[str]):
        from iop4lib.db import RawFit

        try:
            ftp =  ftplib.FTP(cls.ftp_address, encoding=cls.ftp_encoding)
            ftp.login(cls.ftp_user, cls.ftp_password)

            for rawfit in rawfits:

                if isinstance(rawfit, str): # make sure that this is indeed the way tel, night, filepath are built
                    fileloc = rawfit
                    tel, night, filename = RawFit.fileloc_to_tel_night_filename(fileloc)
                    yyyy_mm_dd = night.strftime("%Y-%m-%d")
                    filepath = Path(iop4conf.datadir) / "raw" / tel / yyyy_mm_dd / filename
                elif isinstance(rawfit, RawFit):
                    fileloc = rawfit.fileloc
                    tel, night, filename = rawfit.telescope, rawfit.night, rawfit.filename
                    yyyy_mm_dd = night.strftime("%Y-%m-%d")
                    filepath =  Path(rawfit.filepath)

                rel_path = os.path.join(cls.epochname_to_remote_dirname(f"{tel}/{yyyy_mm_dd}"), filename)

                logger.debug(f"Downloading {fileloc} from {cls.name} archive ...")

                if not filepath.parent.exists():
                    os.makedirs(filepath.parent)

                with open(filepath, 'wb') as f:
                    ftp.retrbinary(f"RETR {rel_path}", f.write)

            ftp.quit()
        except Exception as e:
            raise Exception(f"Error downloading rawfits: {e}.")

    @classmethod
    def list_remote_filelocs(cls, epochnames: list[str]) -> list[str]:
        from iop4lib.db import Epoch

        ftp =  ftplib.FTP(cls.ftp_address, encoding=cls.ftp_encoding)
        ftp.login(cls.ftp_user, cls.ftp_password)

        dirnames = ftp.nlst()

        fileloc_list = list()

        for epochname in epochnames:

            tel, night = Epoch.epochname_to_tel_night(epochname)
            yymmdd = night.strftime("%y%m%d")

            remote_dir = cls.epochname_to_remote_dirname(epochname)

            if remote_dir not in dirnames:
                logger.error(f"{cls.name} remote dir {remote_dir} does not exist.")
                continue

            try:
                fnames = [fpath.split("/")[-1] for fpath in ftp.nlst(f"/{remote_dir}")]
                fileloc_list.extend([f"{epochname}/{fname}" for fname in fnames if cls.re_expr_fnames.search(fname) and fname != '.' and fname != '..'])
        

            except Exception as e:
                logger.error(f"Error listing {cls.name} remote dir for {epochname}: {e}.")
            
        ftp.quit()

        return fileloc_list