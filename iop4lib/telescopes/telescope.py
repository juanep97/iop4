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

# logging
import logging
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from iop4lib.db import RawFit, ReducedFit

class Telescope(metaclass=ABCMeta):
    """
        Inherit this class to provide telescope specific functionality and
        translations.

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

    # telescope properties

    @property
    @abstractmethod
    def gain_e_adu(self):
        pass

    # Abstract methods

    @classmethod
    @abstractmethod
    def list_remote_raw_fnames(cls, epoch):
        pass

    @classmethod
    @abstractmethod
    def download_rawfits(cls, epoch):
        pass

    @classmethod
    @abstractmethod
    def list_remote_epochnames(cls):
        pass

    @classmethod
    @abstractmethod
    def classify_juliandate_rawfit(cls, rawfit):
        pass

    @classmethod
    @abstractmethod
    def classify_imgtype_rawfit(cls, rawfit):
        pass

    @classmethod
    @abstractmethod
    def classify_band_rawfit(cls, rawfit):
        pass

    @classmethod
    @abstractmethod
    def classify_obsmode_rawfit(cls, rawfit):
        pass

    @classmethod
    @abstractmethod
    def get_header_hintcoord(cls, rawfit, *args, **kwargs):
        pass
    
    @classmethod
    @abstractmethod
    def get_astrometry_position_hint(cls, rawfit, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def get_astrometry_size_hint(cls, rawfit):
        pass

    # Not Implemented Methods (skeleton)

    # @classmethod
    # def compute_relative_photometry(cls):
    #     raise NotImplementedError
    
    @classmethod
    def compute_absolute_photometry(cls):
        raise NotImplementedError
    
    @classmethod
    def compute_relative_polarimetry(cls):
        raise NotImplementedError

    # Class methods (usable)
    
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
    

    @classmethod
    def classify_rawfit(cls, rawfit: 'RawFit'):
        cls.check_telescop_kw(rawfit)
        cls.classify_instrument_kw(rawfit)
        cls.classify_juliandate_rawfit(rawfit)
        cls.classify_imgtype_rawfit(rawfit)
        cls.classify_band_rawfit(rawfit)
        cls.classify_obsmode_rawfit(rawfit)
        cls.classify_imgsize(rawfit)
        cls.classify_exptime(rawfit)

    # telescope independent functionality (they don't need to be overriden in subclasses)


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
        
        if instrume_header == "AndorT90":
            rawfit.instrument = INSTRUMENTS.AndorT90
        elif instrume_header == "Andor":
            rawfit.instrument = INSTRUMENTS.AndorT150
        elif instrume_header == "CAFOS 2.2":
            rawfit.instrument = INSTRUMENTS.CAFOS
        else:
            raise ValueError(f"INSTRUME in fits header ({instrume_header}) not known.")
    
    @classmethod
    def classify_imgsize(cls, rawfit):
        import astropy.io.fits as fits
        from iop4lib.db import RawFit

        with fits.open(rawfit.filepath) as hdul:
            if hdul[0].header["NAXIS"] == 2:
                sizeX = hdul[0].header["NAXIS1"]
                sizeY = hdul[0].header["NAXIS2"]
                rawfit.imgsize = f"{sizeX}x{sizeY}"
                return rawfit.imgsize
            else:
                raise ValueError(f"Raw fit file {rawfit.fileloc} has NAXIS != 2, cannot get imgsize.")
            
    @classmethod
    def classify_exptime(cls, rawfit):
        """
        EXPTIME is an standard FITS keyword, measured in seconds.
        """
        import astropy.io.fits as fits
        from iop4lib.db import RawFit

        with fits.open(rawfit.filepath) as hdul:
            rawfit.exptime = hdul[0].header["EXPTIME"]


    @classmethod
    def get_header_objecthint(self, rawfit):
        r""" Get a hint for the AstroSource in this image from the header. OBJECT is a standard keyword. Return None if none found. 
        
        At the moment his only tries to match sources
        with the IAU name format `[0-9]*\+[0-9]*`.
        """
        
        from iop4lib.db import AstroSource

        object_header = rawfit.header["OBJECT"]
        
        matchs = re.findall(r".*?([0-9]*\+[0-9]*).*", object_header)
        if len(matchs) > 0:
            return AstroSource.objects.filter(name__contains=matchs[0]).first()
        else:
            return None




    # should not depend on the telescope

    @classmethod
    def compute_aperture_photometry(cls, redf, aperpix, r_in, r_out):

        from iop4lib.db.aperphotresult import AperPhotResult
        from iop4lib.utils.sourcedetection import get_bkg, get_segmentation
        from photutils.utils import circular_footprint
        from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry
        from photutils.utils import calc_total_error
        from astropy.stats import SigmaClip
        from iop4lib.utils import get_target_fwhm_aperpix

        if redf.mdata.shape[0] == 1024:
            bkg_box_size = 128
        elif redf.mdata.shape[0] == 2048:
            bkg_box_size = 256
        elif redf.mdata.shape[0] == 800:
            bkg_box_size = 100
        else:
            logger.warning(f"Image size {redf.mdata.shape[0]} not expected.")
            bkg_box_size = redf.mdata.shape[0]//10

        bkg = get_bkg(redf.mdata, filter_size=1, box_size=bkg_box_size)
        # img_bkg_sub = redf.mdata - bkg.background
        img_bkg_sub = redf.mdata

        if np.sum(redf.mdata <= 0.0) >= 1:
            logger.debug(f"{redf}: {np.sum(redf.mdata <= 0.0):.0f} px < 0  ({math.sqrt(np.sum(redf.mdata <= 0.0)):.0f} px2) in IMAGE.")

        # if np.sum(img_bkg_sub <= 0.0) >= 1:
        #     try:
        #         logger.debug(f"{redf}: {np.sum(img_bkg_sub <= 0.0)} px < 0 ({math.sqrt(np.sum(img_bkg_sub <= 0.0)):.0f} px2) in BKG-SUBSTRACTED IMG. Check the bkg-substraction method, I'm going to try to mask sources...")
        #         seg_threshold = 3.0 * bkg.background_rms # safer to ensure they are sources
        #         segment_map, convolved_data = get_segmentation(img_bkg_sub, threshold=seg_threshold, fwhm=1, kernel_size=None, npixels=16, deblend=True)
        #         mask = segment_map.make_source_mask(footprint=circular_footprint(radius=6))
        #         bkg = get_bkg(redf.mdata, filter_size=1, box_size=bkg_box_size, mask=mask)
        #         img_bkg_sub = redf.mdata - bkg.background
        #     except Exception as e:
        #         logger.debug(f"{redf}: can not mask sources here... {e}")
        
        if np.sum(img_bkg_sub <= 0.0) >= 1:
            logger.debug(f"{redf}: {np.sum(img_bkg_sub <= 0.0)} px < 0 ({math.sqrt(np.sum(img_bkg_sub <= 0.0)):.0f} px2) in BKG-SUBSTRACTED IMG, after masking.")


        error = calc_total_error(img_bkg_sub, bkg.background_rms, cls.gain_e_adu)

        for astrosource in redf.sources_in_field.all():
            for pairs, wcs in (('O', redf.wcs1), ('E', redf.wcs2)) if redf.with_pairs else (('O',redf.wcs),):

                ap = CircularAperture(astrosource.coord.to_pixel(wcs), r=aperpix)
                annulus = CircularAnnulus(astrosource.coord.to_pixel(wcs), r_in=r_in, r_out=r_out)

                annulus_stats = ApertureStats(redf.mdata, annulus, error=error, sigma_clip=SigmaClip(sigma=5.0, maxiters=10))
                # bkg_stats = ApertureStats(bkg.background, ap, error=error)
                ap_stats = ApertureStats(redf.mdata, ap, error=error)

                # bkg_flux_counts = bkg_stats.sum
                # bkg_flux_counts_err = bkg_stats.sum_err
                bkg_flux_counts = annulus_stats.median*ap_stats.sum_aper_area.value
                bkg_flux_counts_err = annulus_stats.sum_err / annulus_stats.sum_aper_area.value * ap_stats.sum_aper_area.value

                flux_counts = ap_stats.sum - annulus_stats.mean*ap_stats.sum_aper_area.value
                flux_counts_err = ap_stats.sum_err

                AperPhotResult.create(reducedfit=redf, 
                                      astrosource=astrosource, 
                                      aperpix=aperpix, 
                                      pairs=pairs, 
                                      bkg_flux_counts=bkg_flux_counts, bkg_flux_counts_err=bkg_flux_counts_err,
                                      flux_counts=flux_counts, flux_counts_err=flux_counts_err)
    

    @classmethod
    def compute_relative_photometry(cls, redf: 'ReducedFit') -> None:
        
        from iop4lib.db.aperphotresult import AperPhotResult
        from iop4lib.db.photopolresult import PhotoPolResult
        from iop4lib.utils import get_target_fwhm_aperpix

        if redf.obsmode != OBSMODES.PHOTOMETRY:
            raise Exception(f"{redf}: this method is only for plain photometry images.")
        
        target_fwhm, aperpix, r_in, r_out = get_target_fwhm_aperpix([redf], reductionmethod=REDUCTIONMETHODS.RELPHOT)

        if target_fwhm is None:
            logger.error("Could not estimate a target FWHM, aborting relative photometry.")
            return

        # 1. Compute all aperture photometries

        logger.debug(f"{redf}: computing aperture photometries for {redf}.")

        redf.compute_aperture_photometry(aperpix, r_in, r_out)

        # 2. Compute relative polarimetry for each source (uses the computed aperture photometries)

        logger.debug(f"{redf}: computing relative photometry.")

        # 2. Compute the flux in counts and the instrumental magnitude
        
        photopolresult_L = list()
        
        for astrosource in redf.sources_in_field.all():

            result = PhotoPolResult.create(reducedfits=[redf], astrosource=astrosource, reduction=REDUCTIONMETHODS.RELPHOT)

            aperphotresult = AperPhotResult.objects.get(reducedfit=redf, astrosource=astrosource, aperpix=aperpix, pairs="O")

            result.bkg_flux_counts = aperphotresult.bkg_flux_counts
            result.bkg_flux_counts_err = aperphotresult.bkg_flux_counts_err
            result.flux_counts = aperphotresult.flux_counts
            result.flux_counts_err = aperphotresult.flux_counts_err

            # logger.debug(f"{self}: {result.flux_counts=}")

            if result.flux_counts <= 0.0:
                logger.warning(f"{redf}: negative flux counts encountered while relative photometry for {astrosource=} ??!! They will be nans, but maybe we should look into this...")

            result.mag_inst = -2.5 * np.log10(result.flux_counts) # np.nan if result.flux_counts <= 0.0
            result.mag_inst_err = math.fabs(2.5 / math.log(10) / result.flux_counts * result.flux_counts_err)

            # if the source is a calibrator, compute also the zero point
            if result.astrosource.srctype == SRCTYPES.CALIBRATOR:
                result.mag_known = getattr(result.astrosource, f"mag_{redf.band}")
                result.mag_known_err = getattr(result.astrosource, f"mag_{redf.band}_err", None) or 0.0

                if result.mag_known is None:
                    logger.warning(f"Relative Photometry over {redf}: calibrator {result.astrosource} has no magnitude for band {redf.band}.")
                    result.mag_zp = np.nan
                    result.mag_zp_err = np.nan
                else:
                    result.mag_zp = result.mag_known - result.mag_inst
                    result.mag_zp_err = math.sqrt(result.mag_inst_err**2 + result.mag_known_err**2)
            else:
                # if it is not a calibrator, we can not save the COMPUTED zp, it will be computed and the USED zp will be stored.
                result.mag_zp = None
                result.mag_zp_err = None

            result.save()

            photopolresult_L.append(result)

        # 3. Average the zero points

        calib_mag_zp_array = np.array([result.mag_zp or np.nan for result in photopolresult_L if result.astrosource.srctype == SRCTYPES.CALIBRATOR]) # else it fills with None also and the dtype becomes object
        calib_mag_zp_array = calib_mag_zp_array[~np.isnan(calib_mag_zp_array)]

        calib_mag_zp_array_err = np.array([result.mag_zp_err or np.nan for result in photopolresult_L if result.astrosource.srctype == SRCTYPES.CALIBRATOR])
        calib_mag_zp_array_err = calib_mag_zp_array_err[~np.isnan(calib_mag_zp_array_err)]

        if len(calib_mag_zp_array) == 0:
            logger.error(f"{redf}: can not perform relative photometry without any calibrators for this reduced fit. Deleting results.")
            [result.delete() for result in redf.photopolresults.all()]
            return #raise Exception(f"{self}: can not perform relative photometry without any calibrators for this reduced fit.") 

        zp_avg = np.nanmean(calib_mag_zp_array)
        zp_std = np.nanstd(calib_mag_zp_array)

        zp_err = math.sqrt(np.sum(calib_mag_zp_array_err**2)) / len(calib_mag_zp_array_err)
        zp_err = math.sqrt(zp_std**2 + zp_err**2)

        # 4. Compute the calibrated magnitudes

        for result in photopolresult_L:

            if result.astrosource.srctype == SRCTYPES.CALIBRATOR:
                continue

            # save the zp (to be) used
            result.mag_zp = zp_avg
            result.mag_zp_err = zp_err

            # compute the calibrated magnitude
            result.mag = zp_avg + result.mag_inst
            result.mag_err = math.sqrt(result.mag_inst_err**2 + zp_err**2)

            result.save()
        
        # 5. Save the results

        for result in photopolresult_L:
            result.save()

