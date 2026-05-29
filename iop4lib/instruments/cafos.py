# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
import numpy as np
from collections.abc import Iterable
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
import astrometry

# iop4lib imports
from iop4lib.enums import (
    IMGTYPES,
    BANDS,
    OBSMODES,
)
from .instrument import InstrumentHWP
from iop4lib.telescopes import CAHAT220
from iop4lib.utils.polarization import compute_stokes_HWP_fit_rel

# logging
import logging
logger = logging.getLogger(__name__)

import typing
if typing.TYPE_CHECKING:
    from iop4lib.db import ReducedFit

class CAFOS(InstrumentHWP):
        
    name = "CAFOS2.2"
    telescope = CAHAT220.name

    instrument_kw_L = ["CAFOS 2.2"]

    arcsec_per_pix = 0.530
    gain_e_adu = 1.45
    field_width_arcmin = 34.0
    field_height_arcmin = 34.0

    required_masters = ['masterbias', 'masterflat']

    # pre computed pairs distances to use in the astrometric calibrations
    # obtained from calibrated fields
    
    # computed with:
    # > In [1]: qs = ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.BUILT_REDUCED, instrument="CAFOS2.2").all()
    # > In [2]: disp_sign_mean = np.mean([redf.astrometry_info[-1]['seg_disp_sign'] for redf in qs[len(qs)-300:len(qs)-1]], axis=0)
    # > In [3]: disp_sign_std = np.std([redf.astrometry_info[-1]['seg_disp_sign'] for redf in qs[len(qs)-300:len(qs)-1]], axis=0)

    disp_sign_mean, disp_sign_std = np.array([-35.72492116, -0.19719535]), np.array([1.34389, 1.01621491])
    disp_mean, disp_std = np.abs(disp_sign_mean), disp_sign_std

    default_pol_method = compute_stokes_HWP_fit_rel

    rot_angles_required = {0.0, 22.48, 44.98, 67.48}

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
            elif hdul[0].header['IMAGETYP'] == 'dark':
                rawfit.imgtype = IMGTYPES.DARK
            else:
                logger.error(f"Unknown image type for {rawfit.fileloc}.")
                rawfit.imgtype = IMGTYPES.ERROR
                raise ValueError
            
    @classmethod
    def classify_band_rawfit(cls, rawfit):
        """
        Older data (e.g. 2007): INSFLNAM = 'John R' or INSFLNAM = 'Cous R'
        New data (e.g. 2022): INSFLNAM = 'BessellR'

        There are also images in the archive with INSFLNAM = 'John V' and INSFLNAM = 'John I', and INSFLNAM = 'free'
        """

        from iop4lib.db.rawfit import RawFit
        import astropy.io.fits as fits

        if 'INSFLNAM' in rawfit.header:
            if any([rawfit.header['INSFLNAM'] == kw for kw in ['BessellR', 'John R', 'Cous R', 'CousinsR', 'JohnsonR']]):
                rawfit.band = BANDS.R
            elif any([rawfit.header['INSFLNAM'] == kw for kw in ['BessellI', 'John I', 'Cous I', 'CousinsI', 'JohnsonI']]):
                rawfit.band = BANDS.I
            elif any([rawfit.header['INSFLNAM'] == kw for kw in ['BessellV', 'John V', 'Cous V', 'CousinsV', 'JohnsonV']]):
                rawfit.band = BANDS.V
            elif any([rawfit.header['INSFLNAM'] == kw for kw in ['BessellB', 'John B', 'Cous B', 'CousinsB', 'JohnsonB']]):
                rawfit.band = BANDS.B
            elif any([rawfit.header['INSFLNAM'] == kw for kw in ['BessellU', 'John U', 'Cous U', 'CousinsU', 'JohnsonU']]):
                rawfit.band = BANDS.U
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
                logger.debug(f"Probably not important, but {rawfit.fileloc} is BIAS but has polarimetry keywords, does it makes sense?")
        elif rawfit.header['INSTRMOD'] == 'Polarizer' and rawfit.header['INSPOFPI'] == 'FREE':
            rawfit.obsmode = OBSMODES.PHOTOMETRY
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
    def get_astrometry_position_hint(cls, rawfit, n_field_width=1.5, hintsep=None):
        """ Get the position hint from the FITS header as an astrometry.PositionHint object. 

        Parameters
        ----------
            n_field_width: float, optional
                The search radius in units of field width. Default is 1.5.
            hintsep: Quantity, optional
                The search radius in units of degrees.
        """        

        hintcoord = cls.get_header_hintcoord(rawfit)

        if hintsep is None:
            hintsep = n_field_width * u.Quantity("16 arcmin") # 16 arcmin is the full field size of the CAFOS T2.2, our cut is smaller (6.25, 800x800, but the pointing kws might be from anywhere in the full field)

        return astrometry.PositionHint(ra_deg=hintcoord.ra.deg, dec_deg=hintcoord.dec.deg, radius_deg=hintsep.to_value(u.deg))
    
    @classmethod
    def get_astrometry_size_hint(cls, rawfit):
        """ Get the size hint for this telescope / rawfit.

        from http://w3.caha.es/CAHA/Instruments/CAFOS/cafos22.html
        pixel size in arcmin is around : ~0.530 arcsec
        field size (diameter is) 16.0 arcmin (for 2048 pixels)
        it seems that this is for 2048x2048 images, our images are 800x800 but the fitsheader DATASEC
        indicates it is a cut
        """
        
        return astrometry.SizeHint(lower_arcsec_per_pixel=0.95*cls.arcsec_per_pix,  upper_arcsec_per_pixel=1.05*cls.arcsec_per_pix)

    @classmethod
    def has_pairs(cls, fit_instance):
        """ At the moment, CAFOS polarimetry. """
        return (fit_instance.obsmode == OBSMODES.POLARIMETRY)

    @classmethod
    def build_shotgun_params(cls, redf: 'ReducedFit', params_to_try: dict = None):
        from iop4lib.utils.astrometry import build_shotgun_param_combinations

        params = dict()

        params["d_eps"] = [1.0] #[1*np.linalg.norm(cls.disp_std)]
        params["dx_eps"] = [1.0] #[1*cls.disp_std[0]]
        params["dy_eps"] = [1.0] #[1*cls.disp_std[1]]
        params["dx_min"] = [(cls.disp_mean[0] - 5*cls.disp_std[0])]
        params["dx_max"] = [(cls.disp_mean[0] + 5*cls.disp_std[0])]
        params["dy_min"] = [(cls.disp_mean[1] - 5*cls.disp_std[1])]
        params["dy_max"] = [(cls.disp_mean[1] + 5*cls.disp_std[1])]
        params["d_min"] = [np.linalg.norm(cls.disp_mean) - 3*np.linalg.norm(cls.disp_std)]
        params["d_max"] = [np.linalg.norm(cls.disp_mean) + 3*np.linalg.norm(cls.disp_std)]
        params["bins"] = [400]
        params["hist_range"] = [(0,400)]

        if redf.header_hintobject is not None and redf.header_hintobject.name == "1101+384":
            params["bkg_filter_size"] = [3]
            params["bkg_box_size"] = [16]
            params["seg_fwhm"] = [1.0]
            params["npixels"] = [8, 16]
            params["n_rms_seg"] = [3.0, 1.5, 1.2, 1.1, 1.0]

            if redf.exptime < 6:
                params["npixels"] = [8, 4, 6]
                params["bkg_box_size"] = [8,4,16]

        if params_to_try:
            for k, v in params_to_try.items():
                params[k] = v if isinstance(v,Iterable) else [v]

        return build_shotgun_param_combinations(redf, params_to_try=params)

    @classmethod
    def get_instrumental_polarization(cls, reducedfit) -> dict:
        """ Returns the instrumental polarization for to be used for a given reducedfit."""

        instr_pol_dict = {
            'q_inst' :  0,
            'dq_inst':  0,
            'u_inst' :  0,
            'du_inst':  0,
            'CPA'    :  0,
            'dCPA'   :  0,
        }

        return instr_pol_dict
