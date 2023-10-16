# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
import astrometry

# iop4lib imports
from iop4lib.enums import *
from .instrument import Instrument

# logging
import logging
logger = logging.getLogger(__name__)


class DIPOL(Instrument):

    name = "DIPOL"
    instrument_kw = "ASI Camera (1)"
    
    arcsec_per_pix = 0.134

    @classmethod
    def classify_juliandate_rawfit(cls, rawfit):
        """
        DIPOL files have JD keyword
        """
        import astropy.io.fits as fits
        jd = fits.getheader(rawfit.filepath, ext=0)["JD"]
        rawfit.juliandate = jd


    @classmethod
    def classify_imgtype_rawfit(cls, rawfit):
        """
        DIPOL files have IMAGETYP keyword: Light Frame, Bias Frame

        """
        from iop4lib.db.rawfit import RawFit
        import astropy.io.fits as fits

        with fits.open(rawfit.filepath) as hdul:
            if hdul[0].header['IMAGETYP'] == 'Bias Frame':
                rawfit.imgtype = IMGTYPES.BIAS
            elif hdul[0].header['IMAGETYP'] == 'Light Frame':
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
        elif rawfit.header['FILTER'] == "Red":  
            rawfit.band = BANDS.R
        else:
            rawfit.band = BANDS.ERROR
            raise ValueError(f"Unknown FILTER keyword for {rawfit.fileloc}: {rawfit.header['FILTER']}.")
    

    @classmethod
    def classify_obsmode_rawfit(cls, rawfit):
        """
        In OSN Andor Polarimetry, we only have polarimetry for filter R, and it is indicated as R_45, R0, R45, R90 (-45, 0, 45 and 90 degrees). They correspond
        to the different angles of the polarimeter.

        For photometry, the filter keyword willl be simply the letter R, U, etc.

        The values for angles are -45, 0, 45 and 90.

        Lately we have seen "R-45" instead of "R_45", so we have to take care of that too.
        """

        from iop4lib.db.rawfit import RawFit
        import re

        raise NotImplementedError("DIPOL obsmode not implemented yet")

    @classmethod
    def get_astrometry_size_hint(cls, rawfit):
        """ Get the size hint for this telescope / rawfit.

            For DIPOL in OSN-T090, according to preliminary investigation of OSN crew is:
                Las posiciones que he tomado y el ángulo de rotación en cada caso son estos:
                Dec= -10º HA=+3h  rotación=-177.3º
                Zenit rotación=-177.3º
                Dec=+60º HA=-6h rotación=-177.7º
                Dec=+70º HA=+5h rotación=-177.2º

                El campo es de 9.22 x 6.28 arcmin y el tamaño de pixel de 0.134"/pix

                El ángulo de la imagen cambia muy poco entre las posiciones muy separadas del telescopio, y es de 177.5º ± 0.3º
                Así que como mucho se produce un error de ± 0.3º en las imágenes, y el punto cero es de 2.5º.
        """

        return astrometry.SizeHint(lower_arcsec_per_pixel=0.95*cls.arcsec_per_pix, upper_arcsec_per_pixel=1.05*cls.arcsec_per_pix)