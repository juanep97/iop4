# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
from abc import ABCMeta, abstractmethod

import os
import re
import numpy as np
import math
import astropy.io.fits as fits
import astropy.units as u
import itertools
import datetime
import glob
import astrometry
from photutils.centroids import centroid_sources, centroid_2dg, centroid_com

# iop4lib imports
from iop4lib.enums import *
from iop4lib.utils import filter_zero_points, calibrate_photopolresult
from iop4lib.utils.sourcedetection import apply_gaussian_smooth

# logging
import logging
logger = logging.getLogger(__name__)

from iop4lib.typing import *

class Instrument(metaclass=ABCMeta):
    """ Base class for instruments.
    
        Inherit this class to provide instrument specific functionality (e.g. classification of images,
        reduction, etc).

    """

    # Instrument identification (subclasses must implement these)

    @property
    @abstractmethod
    def name(self):
        """ The name of the instrument."""
        raise NotImplementedError

    @property
    @abstractmethod
    def telescope(self):
        """ The telescope this instrument is mounted on."""
        raise NotImplementedError

    @property
    @abstractmethod
    def instrument_kw(self):
        """ The keyword in the FITS header that identifies this instrument."""
        raise NotImplementedError

    # Instrument specific properties (subclasses must implement these)

    @property
    @abstractmethod
    def field_width_arcmin(self):
        """ Field width in arcmin."""
        raise NotImplementedError

    @property
    @abstractmethod
    def arcsec_per_pix(self):
        """ Pixel size in arcseconds per pixel."""
        raise NotImplementedError

    @property
    @abstractmethod
    def gain_e_adu(self):
        """ Gain in e-/ADU. Used to compute the error in aperture photometry."""
        raise NotImplementedError

    @property
    @abstractmethod
    def required_masters(self):
        r""" List of calibration frames needed.

            Cooled CCD cameras will only need `required_masters = ['masterbias', 'masterflat']` in the subclass, since dark current is close to zero. 
            If dark current is not negligible, set `required_masters = ['masterbias', 'masterdark', 'masterflat']` in the subclass.
        """
        raise NotImplementedError

    # Class methods (you should be using these from the Instrument class, not subclasses)

    @classmethod
    def get_known(cls):
        """ Return a list of all known instruments subclasses."""
        from .osn_cameras import RoperT90, AndorT90, AndorT150
        from .cafos import CAFOS
        from .dipol import DIPOL

        return [RoperT90, AndorT90, AndorT150, CAFOS, DIPOL]

    @classmethod
    def by_name(cls, name: str) -> 'Instrument':
        """
        Try to get instrument subclass by name, else raise Exception.
        """
        for instr in Instrument.get_known():
            if instr.name == name:
                return instr
        raise NotImplementedError(f"Instrument {name} not implemented.")
    
    # Common instrument functionality
    # You should be using these from the subclasses already
    # these don't need to be overriden in subclasses, but they can be

    # classification methods

    @classmethod
    def classify_rawfit(cls, rawfit):
        cls.check_instrument_kw(rawfit)
        cls.classify_juliandate_rawfit(rawfit)
        cls.classify_imgtype_rawfit(rawfit)
        cls.classify_band_rawfit(rawfit)
        cls.classify_obsmode_rawfit(rawfit)
        cls.classify_imgsize(rawfit)
        cls.classify_imgbinning(rawfit)
        cls.classify_exptime(rawfit)
    
    @classmethod
    def check_instrument_kw(cls, rawfit):
        """ Check that the instrument keyword is correct. """
        if rawfit.header["INSTRUME"] not in cls.instrument_kw_L:
            raise ValueError(f"Raw fit file {rawfit.fileloc} has INSTRUME not in {cls.instrument_kw_L}.")

    @classmethod
    def classify_imgsize(cls, rawfit):
        """ Read the size of the image from the FITS header, and save it in rawfit.imgsize."""
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
    def classify_imgbinning(cls, rawfit):
        """ Read the binning of the image from the FITS header, and save it in rawfit.imgbinning."""
        import astropy.io.fits as fits
        from iop4lib.db import RawFit

        with fits.open(rawfit.filepath) as hdul:
            header = hdul[0].header
            if header["NAXIS"] == 2:
                binX = header.get("XBINNING") or header.get("CCDBINX")
                binY = header.get("YBINNING") or header.get("CCDBINY")
                if not binX or not binY:
                    raise ValueError(f"Raw fit file {rawfit} binning could not be read from header.")
                rawfit.imgbinning = f"{binX}x{binY}"
                return rawfit.imgbinning
            else:
                raise ValueError(f"Raw fit file {rawfit} has NAXIS != 2!")

    @classmethod
    def get_rawfit_hint_arcsec_per_pix(cls, rawfit: 'RawFit'):

        if not hasattr(cls, 'reference_binning'):
            return cls.arcsec_per_pix

        hdr = rawfit.header

        if hdr["XBINNING"] != hdr["YBINNING"]:
            raise Exception("Different binning in X and Y.")

        return cls.arcsec_per_pix / cls.reference_binning * hdr['XBINNING']

    @classmethod
    def get_rawfit_hint_field_width_arcmin(cls, rawfit: 'RawFit'):
        
        hdr = rawfit.header

        if hdr["XBINNING"] != hdr["YBINNING"]:
            raise Exception("Different binning in X and Y.")
        
        return cls.get_rawfit_hint_arcsec_per_pix(rawfit) * hdr['NAXIS1'] / 60.0

    @classmethod
    def get_binning_independent_px(cls, rawfit: 'RawFit', px):

        if not hasattr(cls, 'reference_binning'):
            return px
        
        return px * cls.reference_binning / rawfit.header['XBINNING']

    @classmethod
    @abstractmethod
    def classify_juliandate_rawfit(cls, rawfit: 'RawFit'):
        raise NotImplementedError

    @classmethod
    def classify_exptime(cls, rawfit):
        """
        EXPTIME is an standard FITS keyword, measured in seconds.
        """
        import astropy.io.fits as fits
        from iop4lib.db import RawFit

        with fits.open(rawfit.filepath) as hdul:
            rawfit.exptime = hdul[0].header["EXPTIME"]

    # reduction methods

    @classmethod
    def _get_header_hintobject_easy(self, rawfit) -> Union['AstroSource', None]:
        r"""Get a hint for the AstroSource in this image from the header. 
        Return None if none found. 
        
        This method only tries to match the OBJECT keyword with the IAU name 
        format (`[0-9]*\+[0-9]*`) to the name of the sources in the DB.
        """
        
        from iop4lib.db import AstroSource

        object_header = rawfit.header["OBJECT"]
        
        matchs = re.findall(r".*?([0-9]*\+[0-9]*).*", object_header)
        if len(matchs) > 0:
            return AstroSource.objects.filter(name__contains=matchs[0]).first()
        else:
            return None
            
    @classmethod
    def _get_header_hintobject_hard(self, rawfit):
        """Get a hint for the AstroSource in this image from the header. 
        Return None if none found.

        This method tries to match the OBJECT keyword with each source in the
        DB, first trying all of them by their main names, then by their other 
        names.
        """

        from iop4lib.db import AstroSource

        catalog = AstroSource.objects.exclude(is_calibrator=True).all()

        pattern = re.compile(r"^([a-zA-Z0-9]{1,3}_[a-zA-Z0-9]+|[a-zA-Z0-9]{4,})(?=_|$)")
        
        obj_kw = rawfit.header['OBJECT']
        
        match = pattern.match(obj_kw)

        def get_invariable_str(s):
            return s.replace(' ', '').replace('-','').replace('+','').replace('_','').upper()

        if match:
            
            search_str = match.group(0)

            for source in catalog:
                if get_invariable_str(search_str) in get_invariable_str(source.name):
                    return source
                
            for source in catalog:
                if not source.other_names_list:
                    continue
                if any([get_invariable_str(search_str) in get_invariable_str(other_name) for other_name in source.other_names_list]):
                    return source
                
        return None
    
    @classmethod
    def get_header_hintobject(cls, rawfit):
        """Get the hint for the AstroSource in this image from the header.
        Return None if none found. 
        
        OBJECT is a standard keyword. This method will try to match the OBJECT 
        keyword, first using the more strict `_get_header_hintobject_easy`,
        and if that fails, using the more relaxed `_get_header_hintobject_hard`.
        """

        return cls._get_header_hintobject_easy(rawfit) or cls._get_header_hintobject_hard(rawfit)
    
    @classmethod
    def get_header_hintcoord(cls, rawfit):
        """ Get the position hint from the FITS header as a coordinate."""
        raise NotImplementedError
    
    @classmethod
    @abstractmethod 
    def get_astrometry_position_hint(cls, rawfit, n_field_width=1.5, hintsep=None) -> astrometry.PositionHint:
        """ Get the position hint from the FITS header as an astrometry.PositionHint object. """        
        raise NotImplementedError

    @classmethod
    @abstractmethod 
    def get_astrometry_size_hint(cls, rawfit) -> astrometry.SizeHint:
        """ Get the size hint for this telescope / rawfit."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def has_pairs(cls, fit_instance: Union['ReducedFit', 'RawFit']) -> bool:
        """ Indicates whether both ordinary and extraordinary sources are present 
        in the file. At the moment, this happens only for CAFOS polarimetry
        """
        raise NotImplementedError

    @classmethod
    def build_shotgun_params(cls, redf: 'ReducedFit', params_to_try: dict = None):
        from iop4lib.utils.astrometry import build_shotgun_param_combinations
        return build_shotgun_param_combinations(redf, params_to_try=params_to_try)
    
    @classmethod
    def build_wcs(cls, reducedfit: 'ReducedFit', params_to_try : dict = None, summary_kwargs : dict = None) -> 'BuildWCSResult':
        """ Build a WCS for a reduced fit from this instrument. 
        
        By default (Instrument class), this will just call the build_wcs_params_shotgun from iop4lib.utils.astrometry.

        Keyword Arguments
        -----------------
        params_to_try : dict, optional
            The parameters to pass to the shotgun_params function.
        summary_kwargs : dict, optional
            build_summary_images : bool, optional
                Whether to build summary images of the process. Default is True.
            with_simbad : bool, default True
                Whether to query and plot a few Simbad sources in the image. Might be useful to 
                check whether the found coordinates are correct. Default is True.
        """

        if summary_kwargs is None:
            summary_kwargs = {'build_summary_images':True, 'with_simbad':True}

        from iop4lib.utils.astrometry import build_wcs_params_shotgun
        param_dicts_L = cls.build_shotgun_params(reducedfit, params_to_try=params_to_try)
        build_wcs_result = build_wcs_params_shotgun(reducedfit, param_dicts_L=param_dicts_L, summary_kwargs=summary_kwargs)

        return build_wcs_result

    @classmethod
    def request_master(cls, rawfit, model, other_epochs=False):
        """ Searchs in the DB and returns an appropiate master bias / flat / dark for this rawfit. 
        
        Notes
        -----

        It takes into account the parameters (band, size, etc) defined in Master' margs_kwL; except 
        for exptime, since master calibration frames with different exptime can be applied.

        By default, it looks for masters in the same epoch, but if other_epochs is set to True, it
        will look for masters in other epochs. If more than one master is found, it returns the
        one from the closest night. It will print a warning even with other_epochs if it is more than 1
        week away from the rawfit epoch.
        
        If no master is found, it returns None.
        """

        from iop4lib.db import RawFit

        rf_vals = RawFit.objects.filter(id=rawfit.id).values().get()
        args = {k:rf_vals[k] for k in rf_vals if k in model.margs_kwL}
        
        args.pop("exptime", None) # exptime might be a building keywords (for flats and darks), but masters with different exptime can be applied
        args["epoch"] = rawfit.epoch # from .values() we only get epoch__id 

        master = model.objects.filter(**args, flags__hasnot=model.FLAGS.IGNORE).first()
        
        if master is None and other_epochs == True:
            args.pop("epoch")

            master_other_epochs = np.array(model.objects.filter(**args, flags__hasnot=model.FLAGS.IGNORE).all())

            if len(master_other_epochs) == 0:
                logger.debug(f"No {model._meta.verbose_name} for {args} in DB, None will be returned.")
                return None
            
            master_other_epochs_jyear = np.array([md.epoch.jyear for md in master_other_epochs])
            master = master_other_epochs[np.argsort(np.abs(master_other_epochs_jyear - rawfit.epoch.jyear))[0]]
            
            if (master.epoch.jyear - rawfit.epoch.jyear) > 7/365:
                logger.warning(f"{model._meta.verbose_name} from epoch {master.epoch} is more than 1 week away from epoch {rawfit.epoch}.")
                        
        return master


    @classmethod
    def associate_masters(cls, reducedfit, **masters_dict):
        """ Associate a masterbias, masterdark and masterflat to this reducedfit."""

        from iop4lib.db import ReducedFit, MasterBias, MasterDark, MasterFlat

        for (attrname, model) in zip(['masterbias', 'masterdark', 'masterflat'], [MasterBias, MasterDark, MasterFlat]):

            if attrname not in cls.required_masters:
                continue
            
            if masters_dict.get(attrname, None) is not None:
                setattr(reducedfit, attrname, masters_dict[attrname])
            else:
                if (master := reducedfit.rawfit.request_master(model)) is not None:
                    setattr(reducedfit, attrname, master)
                else:
                    logger.warning(f"{reducedfit}: {attrname} in this epoch could not be found, attemptying adjacent epochs.")
                    if (master := reducedfit.rawfit.request_master(model, other_epochs=True)) is not None:
                        setattr(reducedfit, attrname, master)
                    else:
                        logger.error(f"{reducedfit}: Could not find any {attrname}, not even in adjacent epochs.")
                        reducedfit.set_flag(ReducedFit.FLAGS.ERROR)

    @classmethod
    def apply_masters(cls, reducedfit):
        """ Apply the associated calibration frames to the raw fit to obtain the reduced fit."""
        
        import astropy.io.fits as fits

        logger.debug(f"{reducedfit}: applying masters")

        rf_data = fits.getdata(reducedfit.rawfit.filepath)
        mb_data = fits.getdata(reducedfit.masterbias.filepath)
        mf_data = fits.getdata(reducedfit.masterflat.filepath)

        if reducedfit.masterdark is not None:
            md_dark = fits.getdata(reducedfit.masterdark.filepath)
        else :
            logger.warning(f"{reducedfit}: no masterdark found, assuming dark current = 0, is this a CCD camera and it's cold?")
            md_dark = 0

        data_new = (rf_data - mb_data - md_dark*reducedfit.rawfit.exptime) / (mf_data)

        header_new = fits.Header()

        if not os.path.exists(os.path.dirname(reducedfit.filepath)):
            logger.debug(f"{reducedfit}: creating directory {os.path.dirname(reducedfit.filepath)}")
            os.makedirs(os.path.dirname(reducedfit.filepath))
        
        fits.writeto(reducedfit.filepath, data_new, header=header_new, overwrite=True)

    @classmethod
    def astrometric_calibration(cls, reducedfit: 'ReducedFit', **build_wcs_kwargs):
        """ Performs astrometric calibration on the reduced fit, giving it the appropriate WCS.

        If the are both ordinary and extraordinary sources in the field, one WCS will be built for each,
        and the will be saved in the first and second extensions of the FITS file.
        """

        # delete old astrometry info
        for fpath in glob.iglob(os.path.join(reducedfit.filedpropdir, "astrometry_*")):
            os.remove(fpath)

        # build the WCS
        build_wcs_result = cls.build_wcs(reducedfit, **build_wcs_kwargs)

        if build_wcs_result.success:

            logger.debug(f"{reducedfit}: saving WCSs to FITS header.")

            wcs1 = build_wcs_result.wcslist[0]

            header = fits.Header()

            header.update(wcs1.to_header(relax=True, key="A"))

            if reducedfit.has_pairs:
                wcs2 = build_wcs_result.wcslist[1]
                header.update(wcs2.to_header(relax=True, key="B"))

            # if available, save also some info about the astrometry solution
            if 'bm' in build_wcs_result.info:
                bm = build_wcs_result.info['bm']
                # adding HIERARCH avoids a warning, they can be accessed without HIERARCH
                header['HIERARCH AS_ARCSEC_PER_PIX'] = bm.scale_arcsec_per_pixel
                header['HIERARCH AS_CENTER_RA_DEG'] = bm.center_ra_deg
                header['HIERARCH AS_CENTER_DEC_DEG'] = bm.center_dec_deg

            # save the header to the file

            with fits.open(reducedfit.filepath, 'update') as hdul:
                hdul[0].header.update(header)

            # Save some extra info (not in the header)

            if not 'date' in build_wcs_result.info:
                build_wcs_result.info['date'] = datetime.datetime.now()

            try:
                if isinstance(reducedfit.astrometry_info, list):
                    reducedfit.astrometry_info = list(itertools.chain(reducedfit.astrometry_info, [build_wcs_result.info]))
                else:
                    reducedfit.astrometry_info = [build_wcs_result.info]
            except NameError:
                reducedfit.astrometry_info = [build_wcs_result.info]

        else:
            raise Exception(f"Could not perform astrometric calibration on {reducedfit}: {build_wcs_result=}")

    @classmethod
    def build_file(cls, reducedfit: 'ReducedFit', **build_wcs_kwargs):
        """ Builds the ReducedFit FITS file.

        Notes
        -----
        The file is built by:
        - applying master calibration frames.
        - astrometrically calibrate the reduced fit, giving it a WCS.
        - find the catalog sources in the field.
        """

        from iop4lib.db import AstroSource, ReducedFit

        logger.debug(f"{reducedfit}: building file")

        reducedfit.unset_flag(ReducedFit.FLAGS.BUILT_REDUCED)

        reducedfit.apply_masters()
        
        logger.debug(f"{reducedfit}: performing astrometric calibration")

        try:
            reducedfit.astrometric_calibration(**build_wcs_kwargs)
        except Exception as e:
            logger.error(f"{reducedfit}: could not perform astrometric calibration on {reducedfit}: {e}")
            reducedfit.set_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY)
            if reducedfit.auto_merge_to_db:
                reducedfit.save()
            raise e
        else:
            logger.debug(f"{reducedfit}: astrometric calibration was successful.")
            reducedfit.unset_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY)

            logger.debug(f"{reducedfit}: searching for sources in field...")
            sources_in_field = AstroSource.get_sources_in_field(fit=reducedfit)
            
            logger.debug(f"{reducedfit}: found {len(sources_in_field)} sources in field.")
            reducedfit.sources_in_field.set(sources_in_field, clear=True)
                
            reducedfit.set_flag(ReducedFit.FLAGS.BUILT_REDUCED)

        if reducedfit.auto_merge_to_db:
            reducedfit.save()

    @classmethod
    def compute_aperture_photometry(cls, redf, aperpix, r_in, r_out):
        """ Common aperture photometry method for all instruments."""

        from iop4lib.db.aperphotresult import AperPhotResult
        from iop4lib.utils.sourcedetection import get_bkg, get_segmentation
        from photutils.utils import circular_footprint
        from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry
        from photutils.utils import calc_total_error
        from astropy.stats import SigmaClip

        img = redf.mdata

        # centroid_sources and fit_fwhm need bkg-substracted data

        # opt 1) estimate bkg using more complex algs
        bkg_box_size = redf.mdata.shape[0]//10
        bkg = get_bkg(redf.mdata, filter_size=1, box_size=bkg_box_size)
        median = bkg.background_median

        # opt 2) estimate bkg using simple sigma clip
        # mean, median, std = sigma_clipped_stats(img, sigma=3.0)

        error = calc_total_error(img, bkg.background_rms, cls.gain_e_adu)

        apres_L = list()

        for astrosource in redf.sources_in_field.all():
            for pairs, wcs in (('O', redf.wcs1), ('E', redf.wcs2)) if redf.has_pairs else (('O',redf.wcs),):
                try:
                    logger.debug(f"{redf}: computing aperture photometry for {astrosource} {pairs}")

                    wcs_px_pos = astrosource.coord.to_pixel(wcs)

                    # check that the wcs px position is within (r_in+r_out)/2 from ther border of the image

                    r_mid = (r_in + r_out) / 2

                    if not (r_mid < wcs_px_pos[0] < img.shape[1] - r_mid and r_mid < wcs_px_pos[1] < img.shape[0] - r_mid):
                        logger.warning(f"{redf}: ({pairs}) image of {astrosource.name} is too close to the border, skipping aperture photometry.")
                        continue

                    # # choose a box size that is somewhat larger than the aperture
                    # # in case of pairs, cap box size so that it is somewhat smaller than the distance between pairs

                    # box_size = math.ceil(1.6 * aperpix)//2 * 2 + 1

                    # if redf.has_pairs:
                    #     box_size_max = math.ceil(np.linalg.norm(Instrument.by_name(redf.instrument).disp_sign_mean))//2 * 2 - 1
                    #     box_size = min(box_size, box_size_max)

                    # correct position using centroid

                    arcsec_px = redf.pixscale.to(u.arcsec/u.pix).value

                    box_size = math.ceil( ( 12 / arcsec_px ) ) // 2 * 2 + 1
                    # if there is some astrometric deviation (e.g. for the extaordinary image), perhaps try 24 arcsec, 12 might be to small for it...

                    logger.debug(f"{box_size=}")

                    if redf.has_pairs:
                        disp_sign_mean = redf.instrument_cls.get_binning_independent_px(redf.rawfit, redf.instrument_cls.disp_sign_mean)
                        box_size_max = math.ceil(np.linalg.norm(disp_sign_mean))//2 * 2 - 1
                        box_size = min(box_size, box_size_max)

                    # centroid_px_pos = centroid_sources(img-median, xpos=wcs_px_pos[0], ypos=wcs_px_pos[1], box_size=box_size, centroid_func=centroid_2dg)
                    # # this should fix when there is a large deviation...:
                    # # centroid_px_pos = centroid_sources(img-median, xpos=wcs_px_pos[0], ypos=wcs_px_pos[1], box_size=box_size, centroid_func=centroid_com)
                    # # centroid_px_pos = centroid_sources(img-median, xpos=centroid_px_pos[0], ypos=centroid_px_pos[1], box_size=box_size, centroid_func=centroid_2dg)
                    # and this should also reduce errors due to diffraction spikes (non-gaussian shape)
                    centroid_px_pos = centroid_sources(img-median, xpos=wcs_px_pos[0], ypos=wcs_px_pos[1], box_size=box_size, centroid_func=centroid_com)
                    centroid_px_pos = centroid_sources(apply_gaussian_smooth(img-median,1), xpos=centroid_px_pos[0], ypos=centroid_px_pos[1], box_size=box_size, centroid_func=centroid_2dg)

                    centroid_px_pos = (centroid_px_pos[0][0], centroid_px_pos[1][0])

                    # log the difference between the WCS and the centroid
                    wcs_diff = np.sqrt((centroid_px_pos[0] - wcs_px_pos[0])**2 + (centroid_px_pos[1] - wcs_px_pos[1])**2)
                    
                    logger.debug(f"ReducedFit {redf.id}: {astrosource.name} {pairs}: WCS centroid distance = {wcs_diff:.1f} px")

                    # check that the centroid position is within the borders of the image

                    if not (r_mid < centroid_px_pos[0] < img.shape[1] - r_mid and r_mid < centroid_px_pos[1] < img.shape[0] - r_mid):
                        logger.warning(f"{redf}: centroid of the ({pairs}) image of {astrosource.name} is too close to the border, skipping aperture photometry.")
                        continue

                    ap = CircularAperture(centroid_px_pos, r=aperpix)
                    annulus = CircularAnnulus(centroid_px_pos, r_in=r_in, r_out=r_out)

                    annulus_stats = ApertureStats(redf.mdata, annulus, error=error, sigma_clip=SigmaClip(sigma=5.0, maxiters=10))
                    ap_stats = ApertureStats(redf.mdata, ap, error=error)

                    bkg_flux_counts = annulus_stats.median*ap_stats.sum_aper_area.value
                    bkg_flux_counts_err = annulus_stats.sum_err / annulus_stats.sum_aper_area.value * ap_stats.sum_aper_area.value

                    flux_counts = ap_stats.sum - annulus_stats.mean*ap_stats.sum_aper_area.value # TODO: check if i should use mean!
                    flux_counts_err = ap_stats.sum_err

                    apres = AperPhotResult.create(
                        reducedfit=redf, 
                        astrosource=astrosource, 
                        aperpix=aperpix, 
                        r_in=r_in, r_out=r_out,
                        x_px=centroid_px_pos[0], y_px=centroid_px_pos[1],
                        pairs=pairs, 
                        bkg_flux_counts=bkg_flux_counts, bkg_flux_counts_err=bkg_flux_counts_err,
                        flux_counts=flux_counts, flux_counts_err=flux_counts_err,
                    )
                    
                    apres_L.append(apres)
                except Exception as e:
                    logger.warning(f"{redf}: error computing aperture photometry for {astrosource} {pairs}: {e}")

        return apres_L
    
    @classmethod
    def compute_relative_photometry(cls, redf: 'ReducedFit') -> None:
        """ Common relative photometry method for all instruments. """
        
        from iop4lib.db.aperphotresult import AperPhotResult
        from iop4lib.db.photopolresult import PhotoPolResult

        if redf.obsmode != OBSMODES.PHOTOMETRY:
            raise Exception(f"{redf}: this method is only for plain photometry images.")
        
        aperpix, r_in, r_out, fit_res_dict = cls.estimate_common_apertures([redf], reductionmethod=REDUCTIONMETHODS.RELPHOT)
        target_fwhm = fit_res_dict['mean_fwhm']
        
        if target_fwhm is None:
            logger.error("Could not estimate a target FWHM, aborting relative photometry.")
            return

        # 1. Compute all aperture photometries

        logger.debug(f"{redf}: computing aperture photometries for {redf} (target_fwhm = {target_fwhm:.1f} px, aperpix = {aperpix:.1f} px, r_in = {r_in:.1f} px, r_out = {r_out:.1f} px).")

        cls.compute_aperture_photometry(redf, aperpix, r_in, r_out)

        # 2. Compute relative polarimetry for each source (uses the computed aperture photometries)

        logger.debug(f"{redf}: computing relative photometry.")

        # 2. Compute the flux in counts and the instrumental magnitude
        
        photopolresult_L = list()
        
        for astrosource in redf.sources_in_field.all():

            qs_aperphotresult = AperPhotResult.objects.filter(reducedfit=redf, astrosource=astrosource, aperpix=aperpix, pairs="O")

            if not qs_aperphotresult.exists():
                logger.error(f"{redf}: no aperture photometry for source {astrosource.name} found, skipping relative photometry.")
                continue

            aperphotresult = qs_aperphotresult.first()

            result = PhotoPolResult.create(reducedfits=[redf], astrosource=astrosource, reduction=REDUCTIONMETHODS.RELPHOT)

            result.aperpix = aperpix
            result.aperas = aperpix * redf.pixscale.to(u.arcsec / u.pix).value
            result.fwhm = target_fwhm * redf.pixscale.to(u.arcsec / u.pix).value
            result.bkg_flux_counts = aperphotresult.bkg_flux_counts
            result.bkg_flux_counts_err = aperphotresult.bkg_flux_counts_err
            result.flux_counts = aperphotresult.flux_counts
            result.flux_counts_err = aperphotresult.flux_counts_err

            # logger.debug(f"{self}: {result.flux_counts=}")

            if result.flux_counts is None: # when does this happen? when there is a source whose apertue falls partially outside the image? https://github.com/juanep97/iop4/issues/24
                logger.error(f"{redf}: during relative photometry, encountered flux_counts=None for source {astrosource.name}, aperphotresult {aperphotresult.id}!!!")
                result.flux_counts = np.nan
                result.flux_counts_err = np.nan

            if result.flux_counts <= 0.0:
                logger.warning(f"{redf}: negative flux counts encountered while relative photometry for {astrosource=} ??!! They will be nans, but maybe we should look into this...")

            result.mag_inst = -2.5 * np.log10(result.flux_counts) # np.nan if result.flux_counts <= 0.0
            result.mag_inst_err = math.fabs(2.5 / math.log(10) / result.flux_counts * result.flux_counts_err)

            # if the source is a calibrator, compute also the zero point
            if result.astrosource.is_calibrator:
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

            result.aperphotresults.set([aperphotresult], clear=True)

            result.save()

            photopolresult_L.append(result)

        # 3. Compute the calibrated magnitudes

        for result in photopolresult_L:

            if result.astrosource.is_calibrator:
                continue

            logger.debug(f"{redf}: calibrating {result}")

            calibrate_photopolresult(result, photopolresult_L)
        
        # 5. Save the results

        for result in photopolresult_L:
            result.save()



from abc import ABC, abstractmethod

class InstrumentHWP(ABC, Instrument):

    @property
    @abstractmethod
    def rot_angles_required(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_instrumental_polarization(redf):
        raise NotImplementedError

    @classmethod
    def compute_relative_polarimetry(cls, polarimetry_group: 'PolarimetryGroup'):
        """ Computes the relative polarimetry for a polarimetry group (CAFOS AND DIPOL)
        
        .. note::
            CAFOS Polarimetry observations are done with a system consisting of a half-wave plate (HW) and a Wollaston prism (P).

            The rotation angle theta_i refers to the angle theta_i between the HW plate and its fast (extraordinary) axes.

            The effect of the HW is to rotate the polarization vector by 2*theta_i, and the effect of the Wollaston prism is to split 
            the beam into two beams polarized in orthogonal directions (ordinary and extraordinary).

            An input polarized beam with direction v will be rotated by HW by 2*theta_i. The O and E fluxes will be the projections of the
            rotated vector onto the ordinary and extraordinary directions of the Wollaston prism (in absolute values since -45 and 45 
            polarization directions are equivalent). A way to write this is:

            fo(theta_i) = abs( <HW(theta_i)v,e_i> ) = abs ( <R(2*theta_i)v,e_i> ), where <,> denotes the scalar product and R is the rotation matrix.

            Therefore the following observed fluxes should be the same (ommiting the abs for clarity):

            fo(0º) = <v,e_1> = <v,R(-90)R(+90)e_1> = <R(90),R(90)e_i> = <HW(45),R(90)e_1> = fe(45º)
            fo(22º) = <HW(22)v,e_1> = <R(45)v,e_1> = <R(90)R(45)v,R(90)e_1> = <R(135)v,e_1> = <HW(67),R(90)e_1> = fe(67º)
            fo(45º) = <HW(45)v,e_1> = <R(90)v,e_1> = <v,R(-90)e_1> = -<v,e_2> = fe(0º)
            fo(67º) = <HW(67)v,e_1> = <R(135)v,e_1> = <R(90)R(45)v,e_1> = <R(45)v,R(-90)e_1> = <HW(22),R(-90)e_1> = fe(22º)

            See https://arxiv.org/pdf/astro-ph/0509153 (doi 10.1086/497581) for the formulas relating these fluxes to 
            the Stokes parameters.

        .. note::
            This rotation angle has a different meaning than for OSN-T090 Polarimetry observations. For them, it is the rotation angle of a polarized filter
            with respect to some reference direction. Therefore we have the equivalencies (again ommiting the abs for clarity):
            
            OSN(45º) = <v,R(45)e_1> = <R(45)v,R(45)R(45)e_1> = <HW(22),R(90)e_1> = fE(22º) = fO(67º)
            OSN(90º) = <v,R(90)e_1> = <R(90)v,R(90)R(90)e_1> = fO(45º)
            OSN(-45º) = OSN(135º) = abs(<v,R(-45)e_1>) = <R(45)v,e_1> = <R(135)v,R(90)e_1> = fE(67º) = fO(22º)
            OSN(0º) = <v,e_1> = <v,e_1> = fO(0º)

        """
        
        from iop4lib.db.aperphotresult import AperPhotResult
        from iop4lib.db.photopolresult import PhotoPolResult

        from iop4lib.utils import (
            get_column_values,
        )

        from iop4lib.utils.polarization import (
            Stokes,
            compute_stokes_HWP_fit_full,
            compute_stokes_HWP_fit_2,
        )

        # Perform some checks on the group

        if not all([reducedfit.instrument == cls.name for reducedfit in polarimetry_group]):
            raise Exception(f"This method is only for {cls.name} images.")

        ## get the band of the group

        bands = [reducedfit.band for reducedfit in polarimetry_group]

        if len(set(bands)) == 1:
            band = bands[0]
        else: # should not happen
            raise Exception(f"Can not compute relative polarimetry for a group with different bands: {bands}")

        ## check obsmodes

        if not all([reducedfit.obsmode == OBSMODES.POLARIMETRY for reducedfit in polarimetry_group]):
            raise Exception(f"This method is only for polarimetry images.")
        
        ## check sources in the fields

        sources_in_field_qs_list = [reducedfit.sources_in_field.all() for reducedfit in polarimetry_group]
        group_sources = set.intersection(*map(set, sources_in_field_qs_list))

        if len(group_sources) == 0:
            logger.error("No common sources in field for all polarimetry groups.")
            return
        
        if group_sources != set.union(*map(set, sources_in_field_qs_list)):
            diff_sources = set.union(*map(set, sources_in_field_qs_list)) - set.intersection(*map(set, sources_in_field_qs_list))
            logger.warning("Sources in field do not match for all polarimetry groups (ReducedFit %s): %s" % (",".join([str(redf.pk) for redf in polarimetry_group]), str(diff_sources)))

        ## check rotation angles

        rot_angles_available = set([redf.rotangle for redf in polarimetry_group])

        if not rot_angles_available.issubset(cls.rot_angles_required):
            logger.warning(f"Rotation angles missing: {cls.rot_angles_required - rot_angles_available}")

        # and if we want to disallow missing rotation angles

        if len(polarimetry_group) != len(cls.rot_angles_required):
            raise Exception(f"Can not compute relative polarimetry for a group with {len(polarimetry_group)} reducedfits, it should be {len(cls.rot_angles_required)}.")

        # 1. Compute all aperture photometries

        aperpix, r_in, r_out, fit_res_dict = cls.estimate_common_apertures(polarimetry_group, reductionmethod=REDUCTIONMETHODS.RELPHOT)
        target_fwhm = fit_res_dict['mean_fwhm']
        
        if target_fwhm is None:
            logger.error("Could not estimate a target FWHM, aborting relative photometry.")
            return
        
        logger.debug(f"Computing aperture photometries for the {len(polarimetry_group)} reducedfits in the group (target_fwhm = {target_fwhm:.1f}, aperpix = {aperpix:.1f}, r_in = {r_in:.1f}, r_out = {r_out:.1f}).")

        for reducedfit in polarimetry_group:
            cls.compute_aperture_photometry(reducedfit, aperpix, r_in, r_out)

        # 2. Compute relative polarimetry for each source (uses the computed aperture photometries)

        logger.debug("Computing relative polarimetry.")

        photopolresult_L = list()

        for astrosource in group_sources:

            if astrosource.calibrates.count() > 0:
                continue

            logger.debug(f"Computing relative polarimetry for {astrosource}.")

            aperphotresults = AperPhotResult.objects.filter(
                reducedfit__in = polarimetry_group,
                astrosource = astrosource,
                aperpix = aperpix,
                flux_counts__isnull = False,
            )

            if len(aperphotresults) == 0:
                logger.error(f"No aperphotresults found for {astrosource}")
                continue

            # if we want to disallow missing images

            if len(aperphotresults) != 2*len(cls.rot_angles_required):
                logger.error(f"There should be {2*len(cls.rot_angles_required)} aperphotresults for each astrosource in the group, there are {len(aperphotresults)} for {astrosource.name}.")
                continue

            values = get_column_values(aperphotresults, ['reducedfit__rotangle', 'flux_counts', 'flux_counts_err', 'pairs'])

            angles_L = list(sorted(set(values['reducedfit__rotangle'])))

            # if we want to disallow missing rotator angles

            if len(angles_L) != len(cls.rot_angles_required):
                logger.warning(f"There should be {len(cls.rot_angles_required)} different angles, there are {len(angles_L)}.")

            # build a dict
            fluxD = {}
            for pair, angle, flux, flux_err in zip(values['pairs'], values['reducedfit__rotangle'], values['flux_counts'], values['flux_counts_err']):
                if pair not in fluxD:
                    fluxD[pair] = {}
                fluxD[pair][angle] = (flux, flux_err)


            theta = np.array(angles_L)

            fO = np.array([(fluxD['O'][angle][0]) for angle in angles_L])
            dfO = np.array([(fluxD['O'][angle][1]) for angle in angles_L])

            fE = np.array([(fluxD['E'][angle][0]) for angle in angles_L])
            dfE = np.array([(fluxD['E'][angle][1]) for angle in angles_L])
            
            # IOP4 astrocalibration atm works the other way, swap them here
            fO, dfO, fE, dfE = fE, dfE, fO, dfO

            stokes_nocorr = compute_stokes_HWP_fit_full(theta, fO=fO, dfO=dfO, fE=fE, dfE=dfE)

            if cls.name == INSTRUMENTS.DIPOL and astrosource.name == "2200+420":
                # 2200+420 O image is not valid because of E image of some other source falls over it
                stokes_nocorr = compute_stokes_HWP_fit_2(theta, fE=fE, dfE=dfE)

            # logger.debug(f"{astrosource.name} stokes_nocorr {stokes_nocorr} -> p = {stokes_nocorr.p:.2f} %, chi = {stokes_nocorr.chi:.1f}º")

            inst_pol_dict = cls.get_instrumental_polarization(reducedfit=polarimetry_group[0])

            stokes = stokes_nocorr.correct(**inst_pol_dict)

            # logger.debug(f"{astrosource.name} stokes {stokes=} -> p = {stokes.p:.2f} %, chi = {stokes.chi:.1f}º")

            flux = stokes.I
            flux_err = stokes.dI

            mag_inst = -2.5 * np.log10(flux)
            mag_inst_err = math.fabs(2.5 / math.log(10) * flux_err / flux)

            # if the source is a calibrator, compute also the zero point

            if astrosource.is_calibrator:

                mag_known = getattr(astrosource, f"mag_{band}")
                mag_known_err = getattr(astrosource, f"mag_{band}_err", None) or 0.0

                if mag_known is None:
                    logger.warning(f"Calibrator {astrosource} has no magnitude for band {band}.")
                    mag_zp = np.nan
                    mag_zp_err = np.nan
                else:
                    mag_zp = mag_known - mag_inst
                    mag_zp_err = math.sqrt(mag_known_err ** 2 + mag_inst_err ** 2)
            else:
                mag_zp = None
                mag_zp_err = None

            # save the results
                    
            result = PhotoPolResult.create(
                reducedfits=polarimetry_group, 
                astrosource=astrosource, 
                reduction=REDUCTIONMETHODS.RELPOL,
                # polarization
                p = stokes.p,
                p_err = stokes.dp,
                chi = stokes.chi,
                chi_err = stokes.dchi,
                # photometry
                mag_inst=mag_inst, mag_inst_err=mag_inst_err, 
                mag_zp=mag_zp, mag_zp_err=mag_zp_err,
                # other info
                flux_counts = flux,
                _q_nocorr = stokes_nocorr.q,
                _u_nocorr = stokes_nocorr.u,
                _p_nocorr = stokes_nocorr.p,
                _chi_nocorr = stokes_nocorr.chi,
                # other info
                aperpix = aperpix,
                aperas = aperpix * polarimetry_group[0].pixscale.to(u.arcsec / u.pix).value,
                fwhm = target_fwhm * polarimetry_group[0].pixscale.to(u.arcsec / u.pix).value,
            )

            result.aperphotresults.set(aperphotresults, clear=True)
                        
            photopolresult_L.append(result)

        if not photopolresult_L:
            logger.error("No results could be computed for this group.")
            return
        
        # 3. Compute the calibrated magnitudes for non-calibrators in the group using the averaged zero point

        for result in photopolresult_L:

            if result.astrosource.is_calibrator:
                continue

            logger.debug(f"calibrating {result}")

            calibrate_photopolresult(result, photopolresult_L)

        # 4. Save results

        for result in photopolresult_L:
            result.save()
