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

# iop4lib imports
from iop4lib.enums import *
from .telescope import Telescope

# logging
import logging
logger = logging.getLogger(__name__)

class CAHAT220(Telescope, metaclass=ABCMeta):
    """
    CAHA T220 telescope.

    CAHA has a per-program ftp login (PI user and password in config).
    The file structure when you login is a list of files as:
    {yymmdd}_CAFOS/
    where CAFOS refers to the polarimeter
    and inside each folder there are the files for that day.
    """

    # telescope identification

    name = "CAHA-T220"
    abbrv = "T220"
    telescop_kw = "CA-2.2"

    # telescope specific properties

    gain_e_adu = 1.45

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

            remote_fnameL = [s for s in remote_fnameL_all if re.compile(r".*\.fits?").search(s)] # Filter by filename pattern (get only our files)
            
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
    def classify_juliandate_rawfit(cls, rawfit):
        """
        CAHA T220 has a DATE keyword in the header in ISO format.
        """
        import astropy.io.fits as fits
        from astropy.time import Time 
        date = fits.getheader(rawfit.filepath, ext=0)["DATE"]
        jd = Time(date, format='isot', scale='utc').jd
        rawfit.juliandate = jd

    @classmethod
    def classify_imgtype_rawfit(cls, rawfit):
        """
        CAHA T220 has a IMAGETYP keyword in the header: flat, bias, science
        """
        from iop4lib.db.rawfit import RawFit
        import astropy.io.fits as fits

        with fits.open(rawfit.filepath) as hdul:
            if hdul[0].header['IMAGETYP'] == 'flat':
                rawfit.imgtype = IMGTYPES.FLAT
            elif hdul[0].header['IMAGETYP'] == 'bias':
                rawfit.imgtype = IMGTYPES.BIAS
            elif hdul[0].header['IMAGETYP'] == 'science':
                rawfit.imgtype = IMGTYPES.LIGHT
            else:
                logger.error(f"Unknown image type for {rawfit.fileloc}.")
                rawfit.imgtype = IMGTYPES.ERROR
                raise ValueError
            
    @classmethod
    def classify_band_rawfit(cls, rawfit):
        """
        INSFLNAM is BesselR ??
        """

        from iop4lib.db.rawfit import RawFit
        import astropy.io.fits as fits

        if 'INSFLNAM' in rawfit.header:
            if rawfit.header['INSFLNAM'] == 'BessellR':
                rawfit.band = BANDS.R
            else:
                logger.error(f"{rawfit}: unknown filter {rawfit.header['INSFLNAM']}.")
                rawfit.band = BANDS.ERROR
                raise ValueError(f"{rawfit}: unknown filter {rawfit.header['INSFLNAM']}.")
        else: 
            rawfit.band = BANDS.ERROR
            raise ValueError(f"{rawfit}: INSFLNAM keyword not present.")

    @classmethod
    def classify_obsmode_rawfit(cls, rawfit):
        """
        For CAHA T220, if we are dealing with polarimetry, we have:
        INSTRMOD:	Polarizer
        INSPOFPI 	Wollaston
        INSPOROT 	0.0, 22.48, 67.48
        
        I HAVE NOT FOUND YET OTHER VALUES THAT ARE NOT THIS, PRINT A WARNING OTHERWISE.
        """
        from iop4lib.db.rawfit import RawFit

        if rawfit.header['INSTRMOD'] == 'Polarizer' and rawfit.header['INSPOFPI'] == 'Wollaston':
            rawfit.obsmode = OBSMODES.POLARIMETRY
            rawfit.rotangle = float(rawfit.header['INSPOROT'])

            if rawfit.imgtype == IMGTYPES.BIAS:
                logger.debug(f"Probabbly not important, but {rawfit.fileloc} is BIAS but has polarimetry keywords, does it makes sense?")
        else:
            logger.error("Not implemented, please check the code.")

    @classmethod
    def get_header_hintcoord(cls, rawfit):
        """ Get the position hint from the FITS header as a coordinate.

        Images from CAFOS T2.2 have RA, DEC in the header, both in degrees.
        """

        hint_coord = SkyCoord(Angle(rawfit.header['RA'], unit=u.deg), Angle(rawfit.header['DEC'], unit=u.deg), frame='icrs')
        return hint_coord
    
    @classmethod
    def get_astrometry_position_hint(cls, rawfit, allsky=False, n_field_width=1.5):
        """ Get the position hint from the FITS header as an astrometry.PositionHint object. """        

        hintcoord = cls.get_header_hintcoord(rawfit)

        if allsky:
            hintsep = 180
        else:
            hintsep = n_field_width * u.Quantity("16 arcmin").to_value(u.deg) # 16 arcmin is the full field size of the CAFOS T2.2, our cut is smaller (6.25, 800x800, but the pointing kws might be from anywhere in the full field)

        return astrometry.PositionHint(ra_deg=hintcoord.ra.deg, dec_deg=hintcoord.dec.deg, radius_deg=hintsep)
    
    @classmethod
    def get_astrometry_size_hint(cls, rawfit):
        """ Get the size hint for this telescope / rawfit.

        from http://w3.caha.es/CAHA/Instruments/CAFOS/cafos22.html
        pixel size in arcmin is around : ~0.530 arcsec
        field size (diameter is) 16.0 arcmin (for 2048 pixels)
        it seems that this is for 2048x2048 images, our images are 800x800 but the fitsheader DATASEC
        indicates it is a cut
        """

        lower_arcsec_per_pixel = 0.523
        upper_arcsec_per_pixel = 0.537
        
        return astrometry.SizeHint(lower_arcsec_per_pixel=lower_arcsec_per_pixel,  upper_arcsec_per_pixel=upper_arcsec_per_pixel)