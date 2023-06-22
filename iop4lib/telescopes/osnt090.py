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

# logging
import logging
logger = logging.getLogger(__name__)

class OSNT090(Telescope, metaclass=ABCMeta):
    
    # telescope identification

    name = "OSN-T090"
    abbrv = "T090"
    telescop_kw = "T90-OSN"

    # telescope specific properties

    gain_e_adu = 4.5

    # telescope specific methods

    @classmethod
    def list_remote_epochnames(cls):
        try:
            logger.debug(f"Loging to OSN FTP server")

            ftp =  ftplib.FTP(iop4conf.osn_address)
            ftp.login(iop4conf.osn_user, iop4conf.osn_password)
            remote_dirnameL_all = ftp.nlst()
            ftp.quit()

            logger.debug(f"Total of {len(remote_dirnameL_all)} dirs in OSN.")

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
            raise Exception(f"Error listing remote epochs in OSN: {e}.")


    @classmethod
    def list_remote_raw_fnames(cls, epoch):
        try:
            logger.debug(f"Loging to OSN FTP server")

            ftp =  ftplib.FTP(iop4conf.osn_address)
            ftp.login(iop4conf.osn_user, iop4conf.osn_password)

            logger.debug(f"Changing to OSN dir {epoch.yyyymmdd}")

            ftp.cwd(f"{epoch.yyyymmdd}")
            remote_fnameL_all = ftp.nlst()
            ftp.quit()

            logger.debug(f"Total of {len(remote_fnameL_all)} files in OSN {epoch.epochname}: {remote_fnameL_all}.")

            remote_fnameL = [s for s in remote_fnameL_all if re.compile('|'.join(iop4conf.osn_fnames_patterns)).search(s)] # Filter by filename pattern (get only our files)
            
            logger.debug(f"Filtered to {len(remote_fnameL)} files in OSN {epoch.epochname}.")

            return remote_fnameL
        
        except Exception as e:
            raise Exception(f"Error listing remote dir for {epoch.epochname}: {e}.")
        
    @classmethod
    def download_rawfits(cls, rawfits):
        try:
            ftp =  ftplib.FTP(iop4conf.osn_address)
            ftp.login(iop4conf.osn_user, iop4conf.osn_password)

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
    def classify_juliandate_rawfit(cls, rawfit):
        """
        OSN-T090 fits has JD keyword
        """
        import astropy.io.fits as fits
        jd = fits.getheader(rawfit.filepath, ext=0)["JD"]
        rawfit.juliandate = jd

    @classmethod
    def classify_imgtype_rawfit(cls, rawfit):
        """
        OSN-T090 fits has IMAGETYP keyword: FLAT, BIAS, LIGHT
        """
        from iop4lib.db.rawfit import RawFit
        import astropy.io.fits as fits

        with fits.open(rawfit.filepath) as hdul:
            if hdul[0].header['IMAGETYP'] == 'FLAT':
                rawfit.imgtype = IMGTYPES.FLAT
            elif hdul[0].header['IMAGETYP'] == 'BIAS':
                rawfit.imgtype = IMGTYPES.BIAS
            elif hdul[0].header['IMAGETYP'] == 'LIGHT':
                rawfit.imgtype = IMGTYPES.LIGHT
            else:
                logger.error(f"Unknown image type for {rawfit.fileloc}.")
                rawfit.imgtype = IMGTYPES.ERROR
                raise ValueError

    @classmethod
    def classify_band_rawfit(cls, rawfit):
        """
            OSN Files have no FILTER keyword if they are BIAS, FILTER=Clear if they are FLAT, and FILTER=FilterName if they are LIGHT.
            For our DB, we have R, U, ..., None, ERROR.

            For polarimetry, which is done by taking four images with the R filter at different angles, we have R_45, R0, R45, R90.
        """

        from iop4lib.db.rawfit import RawFit

        if 'FILTER' not in rawfit.header:
            if rawfit.imgtype == IMGTYPES.BIAS:
                rawfit.band = BANDS.NONE
            else:
                rawfit.band = BANDS.ERROR
                raise ValueError(f"Missing FILTER keyword for {rawfit.fileloc} which is not a bias (it is a {rawfit.imgtype}).")
        elif rawfit.header['FILTER'] in {"Clear", ""}:  
            if rawfit.imgtype == IMGTYPES.FLAT:
                rawfit.band = BANDS.NONE
            else:
                rawfit.band = BANDS.ERROR
                raise ValueError(f"FILTER keyword is 'Clear' for {rawfit.fileloc} which is not a flat (it is a {rawfit.imgtype}).")
        else:
            rawfit.band = rawfit.header['FILTER'][0] # First letter of the filter name (R, U, ...) includes cases as R45, R_45, etc   

    @classmethod
    def classify_obsmode_rawfit(cls, rawfit):
        """
        In OSN, we only have polarimetry for filter R, and it is indicated as R_45, R0, R45, R90 (-45, 0, 45 and 90 degrees). They correspond
        to the different angles of the polarimeter.

        For photometry, the filter keyword willl be simply the letter R, U, etc.

        The values for angles are -45, 0, 45 and 90.
        """

        from iop4lib.db.rawfit import RawFit
        import re

        if rawfit.band == BANDS.ERROR:
            raise ValueError("Cannot classify obsmode if band is ERROR.")
        
        if rawfit.band == BANDS.R:
            if rawfit.header['FILTER'] == "R":
                rawfit.obsmode = OBSMODES.PHOTOMETRY
            else:
                logger.debug("Band is R, but FILTER is not exactly R, for OSN this must mean it is polarimetry. Trying to extract angle from FILTER keyword.")

                rawfit.obsmode = OBSMODES.POLARIMETRY

                if rawfit.header['FILTER'] == "R_45":
                    rawfit.rotangle = -45
                elif rawfit.header['FILTER'] == "R0":
                    rawfit.rotangle = 0
                elif rawfit.header['FILTER'] == "R45":
                    rawfit.rotangle = 45
                elif rawfit.header['FILTER'] == "R90":
                    rawfit.rotangle = 90
                else:
                    raise ValueError(f"Cannot extract angle from FILTER keyword '{rawfit.header['FILTER']}'.")
        else:
            logger.debug("Band is not R, assuming it is photometry.")
            rawfit.obsmode = OBSMODES.PHOTOMETRY

    @classmethod
    def get_header_hintcoord(cls, rawfit):
        """ Get the position hint from the fits header as a SkyCoord.

        OSN T090 / AndorT090 have keywords OBJECT, OBJECTRA, OBJECTDEC in the header; e.g: 
        OBJECT 	TXS0506
        OBJCTRA 	05 09 20 ---> this can be input with unit u.hourangle
        OBJCTDEC 	+05 41 16 ---> this can be input with unit u.deg
        """     
        
        hint_coord = SkyCoord(Angle(rawfit.header['OBJCTRA'], unit=u.hourangle), Angle(rawfit.header['OBJCTDEC'], unit=u.degree), frame='icrs')
        return hint_coord
    
    @classmethod
    def get_astrometry_position_hint(cls, rawfit, allsky=False, n_field_width=1.5):
        """ Get the position hint from the FITS header as an astrometry.PositionHint."""        

        hintcoord = cls.get_header_hintcoord(rawfit)
        
        if allsky:
            hintsep = 180.0
        else:
            hintsep = n_field_width * u.Quantity("13.2 arcmin").to_value(u.deg) # should be a bit less than the field size of AndorT090

        return astrometry.PositionHint(ra_deg=hintcoord.ra.deg, dec_deg=hintcoord.dec.deg, radius_deg=hintsep)
    
    @classmethod
    def get_astrometry_size_hint(cls, rawfit):
        """ Get the size hint for this telescope / rawfit.

            According to OSN T090 camera information (https://www.osn.iaa.csic.es/page/camaras-ccdt150-y-ccdt90) 
            the camera pixels are 0.387as/px and it has a field of view of 13,20' x 13,20'. So we provide close values 
            for the hint. If the files are 1x1 it will be 0.387as/px, if 2x2 it will be twice.
        """

        if rawfit.header['NAXIS1'] == 2048:
            return astrometry.SizeHint(lower_arcsec_per_pixel=0.380, upper_arcsec_per_pixel=0.394)
        elif rawfit.header['NAXIS1'] == 1024:
            return astrometry.SizeHint(lower_arcsec_per_pixel=2*0.380, upper_arcsec_per_pixel=2*0.394)
        
