# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
from abc import ABC, ABCMeta, abstractmethod

import os
import re
import numpy as np
import math
import astropy.io.fits as fits
import astropy.units as u
import itertools
import datetime
import glob
import warnings
import traceback

import astrometry
from photutils.centroids import centroid_sources, centroid_2dg, centroid_com
from astropy.nddata import Cutout2D
from astropy.utils.exceptions import AstropyUserWarning
from photutils.psf import fit_fwhm

# iop4lib imports
from iop4lib.enums import (
    OBSMODES,
    REDUCTIONMETHODS,
)
from iop4lib.utils import (
    overlaps_border,
    next_odd,
    get_column_values,
)
from iop4lib.utils.sourcedetection import (
    apply_gaussian_smooth,
    get_bkg,
    get_segmentation,
    get_cat_sources_from_segment_map,
    get_bkg,
    mask_other_sources,
)
from iop4lib.utils.photometry import (
    calibrate_photopolresult,
    NoCalibratorsFound,
)
from iop4lib.utils.polarization import (
    compute_stokes_HWP_fit_1pair,
)
     
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
        
        args.pop("exptime", None) # exptime might be a building keyword (for flats and darks), but masters with different exptime can be applied
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
            sources_in_field = AstroSource.get_sources_in_field(redf=reducedfit)
            
            logger.debug(f"{reducedfit}: found {len(sources_in_field)} sources in field.")
            reducedfit.sources_in_field.set(sources_in_field, clear=True)
                
            reducedfit.set_flag(ReducedFit.FLAGS.BUILT_REDUCED)

        if reducedfit.auto_merge_to_db:
            reducedfit.save()


    @classmethod
    def get_centroids_and_fwhms(cls, redf, use_cutout=True, fwhm_stats=None) -> 'CentroidsAndFwhmResultTuple':

        logger.info(f"Computing centroids and fwhms for {redf}")

        img = redf.mdata

        bkg_box_size = img.shape[0]//10
        bkg = get_bkg(img, filter_size=1, box_size=bkg_box_size)
        
        # data = img - bkg.background_median
        data = img - bkg.background

        # --- First, estimate fwhm for the image using source detection on the image
        # (if the stats are not given, already)

        if fwhm_stats is None:

            n_seg_threshold = 4 # use 3 for more stats (but 4 is faster, less sources)
            npixels = 6 # use 4 for more stats (but 6 is faster, less sources)
            seg_threshold = n_seg_threshold * bkg.background_rms
            segment_map, convolved_data = get_segmentation(data, fwhm=1, npixels=npixels, threshold=seg_threshold)
            seg_cat, positions, tb = get_cat_sources_from_segment_map(segment_map, data, convolved_data)

            logger.debug(f"{len(seg_cat)=}")

            with u.set_enabled_equivalencies(redf.pixscale_equiv):

                fwhm_init = (3*u.arcsec).to_value('pix')

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=AstropyUserWarning)
                    fwhms = fit_fwhm(data, xypos=positions, fwhm=fwhm_init, fit_shape=next_odd(3*fwhm_init))

                fwhms = fwhms * u.pix

                detected_fwhms = fwhms

                fwhm_mean = np.nanmean(fwhms)
                fwhm_median = np.nanmedian(fwhms)
                fwhm_std = np.nanstd(fwhms)

                logger.debug("FWHM results:")
                logger.debug(f"  mean: {fwhm_mean:.1f} ({fwhm_mean.to('arcsec'):.1f})")
                logger.debug(f"  median: {fwhm_median:.1f} ({fwhm_median.to('arcsec'):.1f})")
                logger.debug(f"  std: {fwhm_std:.1f} ({fwhm_std.to('arcsec'):.1f}")

            fwhm_stats = FwhmStatsTuple(fwhm_mean, fwhm_std, fwhm_median)

        else:
            
            fwhm_median = fwhm_stats.median
            detected_fwhms = None

        # --- Then, compute source centroids and fwhm for each of our catalog sources

        centroids_and_fwhms_dict = dict()

        with u.set_enabled_equivalencies(redf.pixscale_equiv):

            if use_cutout:
                # cutout_size = 10 * fwhm_median.to_value('pix')
                cutout_size = next_odd((12*u.arcsec).to_value('pix')) 
                logger.debug(f"cutout_size = {cutout_size:.1f}")
            
            for astrosource in redf.sources_in_field.all():

                try:
                    for pairs, wcs in (('O', redf.wcs1), ('E', redf.wcs2)) if redf.has_pairs else (('O',redf.wcs),):                       

                        logger.debug(f"fitting fwhm for ReducedFit {redf.pk} {astrosource.name} pair {pairs}")

                        wcs_pos = astrosource.coord.to_pixel(wcs)

                        wcs_pos_orig = wcs_pos

                        if use_cutout:
                            cutout = Cutout2D(img - bkg.background_median, position=wcs_pos, size=cutout_size, wcs=wcs)
                            data = cutout.data
                            if overlaps_border(*cutout.position_original, *cutout.shape, *img.shape):
                                raise Exception(f"{redf} {astrosource} {pairs}: cutout overlaps with image border, skipping")
                            wcs_pos = cutout.to_cutout_position(wcs_pos)

                        logger.debug(f"wcs_pos = {wcs_pos}")

                        box_size = next_odd((12*u.arcsec).to_value('pix'))

                        logger.debug(f"box_size = {box_size:.1f}")

                        if redf.has_pairs:
                            box_size_max = next_odd(np.linalg.norm(redf.hint_disp_sign_mean))
                            box_size = min(box_size, box_size_max)

                            logger.debug(f"has pairs = True -> clipped to box_size_max = {box_size_max:.1f}")

                        if use_cutout:
                            if box_size > cutout_size:
                                box_size = cutout_size

                                logger.debug("working inside a cutout, clipped box_size to cutout_size")

                        logger.debug(f"box_size = {box_size:.1f}")

                        if overlaps_border(wcs_pos_orig[1], wcs_pos_orig[0], box_size, box_size, *img.shape):
                            raise Exception(f"{redf} {astrosource} {pairs}: box overlaps with image border, skipping")

                        logger.debug("fitting with centroid_com")

                        c_com = centroid_sources(data, xpos=wcs_pos[0], ypos=wcs_pos[1], box_size=box_size, centroid_func=centroid_com)
                        c_com = np.array(c_com).reshape(2)

                        logger.debug(f"c_com = {c_com}")

                        # attempt to fit 2dg to c_com, then to wcs_pos, and if neither works, default to wcs_pos

                        c2 = wcs_pos

                        for pos1, label in [(c_com, 'c_com'), (wcs_pos, 'wcs_pos')]:

                            logger.debug(f"fitting centroid_2dg with {label}")

                            with warnings.catch_warnings(record=True) as fit_warnings:
                                # sigma_kernel = next_odd(box_size/2)
                                sigma_kernel = next_odd(fwhm_median.to_value('pix')/2)
                                data_smooth = apply_gaussian_smooth(data, sigma_kernel)
                                c_2dg = centroid_sources(data_smooth, xpos=pos1[0], ypos=pos1[1], box_size=box_size, centroid_func=centroid_2dg)
                                c_2dg = np.array(c_2dg).reshape(2)
                            
                            logger.debug(f"c_2dg = {c_2dg}")

                            if fit_warnings:
                                logger.warning("centroid_2dg emitted warnings, might have failed:")
                                for warning in fit_warnings:
                                    warnings.warn(str(warning.message), warning.category)
                                continue

                            jumped_outside = not np.all((0 < c_2dg) & (c_2dg < box_size))
                            
                            if jumped_outside:
                                logger.warning("centroid_2dg jumped outside fit box")
                                continue

                            c2 = c_2dg
                            break

                        logger.debug(f"c2 = {c2}")

                        logger.debug("fitting fwhm to c2")

                        with warnings.catch_warnings(record=True) as fit_warnings:
                            fwhm = fit_fwhm(data, xypos=c2, fit_shape=next_odd(box_size), fwhm=fwhm_median.to_value('pix')).item()

                        # # Another of way of catching any warning could be forcing the trigger of any warning as exception
                        # with warnings.catch_warnings(record=True) as w:
                        #     warnings.simplefilter("error")
                        #     try:
                        #         # code that may raise warning-as-error
                        #         ...
                        #     except AstropyUserWarning as e:
                        #         ...

                        logger.debug(f"fwhm = {fwhm:.1f}")

                        if fit_warnings:
                            logger.warning("fit_fwhm emitted warnings:")
                            for warning in fit_warnings:
                                warnings.warn(str(warning.message), warning.category)
                        
                        valid_fwhm = bool(fwhm < box_size)

                        if not valid_fwhm:
                            logger.warning("fwhm is invalid (>box_size), skipping")
                            raise Exception(f"{redf} {astrosource} {pairs}: fit_fwhm failed")

                        fwhm = fwhm * u.pix

                        if use_cutout:
                            c2 = cutout.to_original_position(c2)

                        # TODO
                        # fwhm should smaller than box_size to be valid...
                        # perhaps try to fit_fwhm in c2 then in wcs_pos
                        # like we did above for c2 itself
                            
                        centroids_and_fwhms_dict[(astrosource, pairs)] = CentroidFwhmTuple(c2, fwhm)
                    
                except Exception as e:
                    msg = f"{redf} ({astrosource} {pairs}: error fitting centroid and fwhm: {e}"
                    if not astrosource.is_calibrator:
                        logger.exception(msg)
                    else:
                        logger.debug(msg)

        return CentroidsAndFwhmResultTuple(centroids_and_fwhms_dict, fwhm_stats, detected_fwhms)

    @classmethod
    def estimate_common_apertures(cls, reducedfits: 'PolarimetryGroup', **kwargs) -> CommonAperturesTuple:
        r"""Estimate an appropriate common aperture for a list of reduced fits. Results are in pixels."""

        fwhm_max = kwargs.get('fwhm_max', 15*u.arcsec)
        
        centroids_and_fwhms: dict['ReducedFit', CentroidsAndFwhmResultTuple] = dict()

        # # a)
        # for redf in reducedfits:
        #     centroids_and_fwhms[redf] = cls.get_centroids_and_fwhms(redf)

        # b)
        # speed it up by
        # 1) trying not to compute centroids and fwhms twice for the same redf, 
        # and attaching results as an attribute to redf for next time (useful if
        # f.estimate_common_apertures is called twice on the same redf).
        # and 2) passing the previous fwhm stats (if >1 images) (speeds up
        # polarimetry groups, which should have the ~same median fwhm).
            
        prev_stats = None
        for i, redf in enumerate(reducedfits):
            redf_centroids_and_fwhms = getattr(redf, 'centroids_and_fwhms', None)
            if not redf_centroids_and_fwhms:
                redf_centroids_and_fwhms = cls.get_centroids_and_fwhms(redf, fwhm_stats=prev_stats)
                redf.centroids_and_fwhms = redf_centroids_and_fwhms
                prev_stats = redf_centroids_and_fwhms.fwhm_stats
            centroids_and_fwhms[redf] = redf_centroids_and_fwhms

        # compute aggregate fwhm stats

        detected_fwhms = [centroids_and_fwhms[redf].detected_fwhms for redf in reducedfits]
        detected_fwhms = [x for x in detected_fwhms if x is not None]
        detected_fwhms = list(itertools.chain.from_iterable(detected_fwhms))
        detected_fwhms = u.Quantity(detected_fwhms)

        fwhm_mean = np.nanmean(detected_fwhms)
        fwhm_std = np.nanstd(detected_fwhms)
        fwhm_median = np.nanmedian(detected_fwhms)

        fwhm_stats = FwhmStatsTuple(fwhm_mean, fwhm_std, fwhm_median)
        
        logger.debug(f"Median FWHM (all detected sources) of {reducedfits}: {fwhm_median:.1f}")

        targets__pks = {src.pk for redf in reducedfits for src in redf.sources_in_field.filter(is_calibrator=False).all()}

        target_fwhm_list = list()

        for redf in reducedfits:
            for (src, pair), (c, fwhm) in centroids_and_fwhms[redf].centroids_and_fwhms.items():
                if src.pk in targets__pks:
                    target_fwhm_list.append(fwhm)
                    
        targets_fwhm_median = np.nanmedian(u.Quantity(target_fwhm_list))

        logger.debug(f"Median FWHM (targets only) of {reducedfits}: {targets_fwhm_median:.1f}")

        # ap_fwhm = fwhm_median
        # ap_fwhm = targets_fwhm_median

        # Better: take the max, bc sometimes our targets has a slightly higher 
        # fwhm, and we prefer to overestimate than underestimate (flux might be
        # left out otherwise). However, it might be NaN.
        ap_fwhm = max(fwhm_median, targets_fwhm_median) if not np.isnan(targets_fwhm_median) else fwhm_median
    
        with u.set_enabled_equivalencies(reducedfits[0].pixscale_equiv):
            
            if ap_fwhm > fwhm_max:
                logger.warning(f"aperture FWHM is too big, using fwhm_max {fwhm_max:.1f}")
                ap_fwhm = fwhm_max

            ap_sigma = ap_fwhm / (2*np.sqrt(2*math.log(2)))

            # if reducedfits[0].has_pairs:
            #     disp = np.linalg.norm(reducedfits[0].hint_disp_sign_mean)*u.pix
            #     disp = disp.to(ap_sigma.unit)
            #     if 3*ap_sigma > 0.8*disp: # TODO: needed? 3x, or x5?
            #         logger.warning("aperpix would be larger than 0.8*disp, it will be clipped")
            #         ap_sigma = 0.8*disp/3

        # r_ap, r_in, r_out = 3.0*ap_sigma, 5.0*ap_sigma, 7.0*ap_sigma
        # TODO: better?
        # r_ap, r_in, r_out = 5.0*ap_sigma, 7.0*ap_sigma, 12.0*ap_sigma
        r_ap, r_in, r_out = 3.0*ap_sigma, 5.0*ap_sigma, 9.0*ap_sigma

        return CommonAperturesTuple(r_ap, r_in, r_out, ap_fwhm, fwhm_stats, centroids_and_fwhms)
    

    @classmethod
    def compute_aperture_photometry(cls, redf, common_apertures: 'CommonAperturesTuple') -> List['AperPhotResult']:
        """ Common aperture photometry method for all instruments."""
        from iop4lib.utils import overlaps_border
        from iop4lib.db.aperphotresult import AperPhotResult
        from iop4lib.utils.sourcedetection import get_bkg
        from photutils.aperture import (
            CircularAperture,
            CircularAnnulus,
            ApertureStats,
        )
        from photutils.utils import calc_total_error
        from astropy.stats import SigmaClip

        with u.set_enabled_equivalencies(redf.pixscale_equiv):
            r_ap = common_apertures.r_ap.to_value('pix')
            r_in = common_apertures.r_in.to_value('pix')
            r_out = common_apertures.r_out.to_value('pix')

        img = redf.mdata

        bkg_box_size = img.shape[0]//10
        bkg = get_bkg(img, filter_size=1, box_size=bkg_box_size)
        error = calc_total_error(img, bkg.background_rms, cls.gain_e_adu)

        centroids_and_fwhms_result = common_apertures.centroids_and_fwhms[redf]

        centroids_and_fwhms = centroids_and_fwhms_result.centroids_and_fwhms

        fwhm_stats = centroids_and_fwhms_result.fwhm_stats
        fwhm_median_px = fwhm_stats.median.to_value('pix', equivalencies=redf.pixscale_equiv)
        fwhm_median_as = fwhm_stats.median.to_value('arcsec', equivalencies=redf.pixscale_equiv)

        aperphotresults: List[AperPhotResult] = list()
        
        for astrosource in redf.sources_in_field.all():
            
            for pair in ('O', 'E') if redf.has_pairs else ('O',):
                try:
                    logger.debug(f"{redf}: computing aperture photometry for {astrosource} {pair}")

                    # get centroid position

                    if (astrosource, pair) in centroids_and_fwhms:
                        centroid, source_fwhm = centroids_and_fwhms[(astrosource, pair)]
                    else:
                        logger.warning(f"{redf} could not get centroid for {astrosource.name} {pair}, skipping aperture photometry")
                        continue

                    # check that the centroid position is within the borders of the image
                
                    if overlaps_border(*(centroid[0], centroid[1]), *(r_out, r_out), *img.shape):
                        logger.warning(f"{redf}: {astrosource.name}, ({pair}) is too close to the border, skipping aperture photometry.")
                        continue

                    source_fwhm_as = source_fwhm.to_value('arcsec', equivalencies=redf.pixscale_equiv)

                    ap = CircularAperture(centroid, r=r_ap)
                    annulus = CircularAnnulus(centroid, r_in=r_in, r_out=r_out)

                    if redf.has_pairs:
                        other_pair = 'E' if pair == 'O' else 'O'
                        other_c, _ = centroids_and_fwhms[(astrosource, other_pair)]
                        other_mask = CircularAperture(other_c, r=r_ap).to_mask().to_image(img.shape).astype(bool)
                    else:
                        other_mask = None

                    if astrosource.metadata.get("mask_other_sources", False):

                        detected_mask, _ = mask_other_sources(img, r_ap, fwhm_median_px, exclude=[centroid])

                        if detected_mask is not None:
                            if other_mask is not None:
                                other_mask = other_mask | detected_mask
                            else:
                                other_mask = detected_mask

                    ap_stats = ApertureStats(img, ap, error=error)
                    annulus_stats = ApertureStats(img, annulus, error=error, sigma_clip=SigmaClip(sigma=3.0, maxiters=5), mask=other_mask)

                    bkg_flux_counts = annulus_stats.mean*ap_stats.sum_aper_area.value
                    bkg_flux_counts_err = annulus_stats.sum_err / annulus_stats.sum_aper_area.value * ap_stats.sum_aper_area.value

                    flux_counts = ap_stats.sum - annulus_stats.mean*ap_stats.sum_aper_area.value
                    flux_counts_err = ap_stats.sum_err

                    apres = AperPhotResult.create(
                        reducedfit = redf, 
                        astrosource = astrosource, 
                        aperpix = r_ap, 
                        r_in = r_in,
                        r_out = r_out,
                        pairs = pair,
                        # photometry results
                        bkg_flux_counts = bkg_flux_counts,
                        bkg_flux_counts_err = bkg_flux_counts_err,
                        flux_counts = flux_counts,
                        flux_counts_err = flux_counts_err,
                        # other info
                        x_px = centroid[0],
                        y_px = centroid[1],
                        fwhm = fwhm_median_as,
                        fwhm_source = source_fwhm_as,
                    )
                    
                    aperphotresults.append(apres)
                except Exception as e:
                    logger.error(f"{redf}: error computing aperture photometry for {astrosource} {pair}: {e}")
                    logger.debug(traceback.format_exc())

        return aperphotresults
    
    @classmethod
    def compute_relative_photometry(cls, redf: 'ReducedFit') -> List['PhotoPolResult']:
        """ Common relative photometry method for all instruments. """
        
        from iop4lib.db.aperphotresult import AperPhotResult
        from iop4lib.db.photopolresult import PhotoPolResult

        if redf.obsmode != OBSMODES.PHOTOMETRY:
            raise Exception(f"{redf}: this method is only for plain photometry images.")
        
        common_aps = cls.estimate_common_apertures([redf], reductionmethod=REDUCTIONMETHODS.RELPHOT)

        r_ap = common_aps.r_ap
        r_in = common_aps.r_in
        r_out = common_aps.r_out
        
        fwhm_median  = common_aps.fwhm_stats.median

        centroids_and_fwhms_result = common_aps.centroids_and_fwhms[redf]

        # 1. Compute all aperture photometries

        common_apertures = cls.estimate_common_apertures([redf], reductionmethod=REDUCTIONMETHODS.RELPHOT)

        r_ap = common_apertures.r_ap
        r_in = common_apertures.r_in
        r_out = common_apertures.r_out

        ap_fwhm = common_apertures.ap_fwhm

        fwhm_median = common_apertures.fwhm_stats.median     

        with u.set_enabled_equivalencies(redf.pixscale_equiv):
            aperpix = r_ap.to_value('pix')
            r_in_px = r_in.to_value('pix')
            r_out_px = r_out.to_value('pix')
            aperas = r_ap.to_value('arcsec')
            median_fwhm_as = fwhm_median.to_value('arcsec')

        logger.debug(f"Computing aperture photometries for {redf} (ap_fwhm = {ap_fwhm:.1f}, r_ap = {r_ap:.1f}, r_in = {r_in:.1f}, r_out = {r_out:.1f}).")

        aperphotresults = cls.compute_aperture_photometry(redf, common_apertures)

        aperphotresult_pks = [aper.pk for aper in aperphotresults]

        # 2. Compute relative polarimetry for each source (uses the computed aperture photometries)

        logger.debug(f"{redf}: computing relative photometry.")

        # 2. Compute the flux in counts and the instrumental magnitude
        
        photopolresults: List[PhotoPolResult] = list()
        
        for astrosource in redf.sources_in_field.all():

            aperphotresult_qs = AperPhotResult.objects.filter(
                pk__in=aperphotresult_pks,
            ).filter(
                reducedfit = redf,
                astrosource = astrosource,
                aperpix = aperpix,
                r_in = r_in_px,
                r_out = r_out_px,
                pairs = "O",
            ).all()

            if not aperphotresult_qs.exists():
                logger.error(f"{redf}: no aperture photometry for source {astrosource.name} found, skipping relative photometry.")
                continue

            aperphotresult = aperphotresult_qs.get()

            result = PhotoPolResult.create(reducedfits=[redf], astrosource=astrosource, reduction=REDUCTIONMETHODS.RELPHOT)

            result.aperpix = aperpix
            result.aperas = aperas
            result.fwhm = median_fwhm_as

            with u.set_enabled_equivalencies(redf.pixscale_equiv):
                fwhm_source = centroids_and_fwhms_result.centroids_and_fwhms[(astrosource, 'O')].fwhm
                fwhm_source_as = fwhm_source.to_value('arcsec')

            result.fwhm_source = fwhm_source_as

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

            photopolresults.append(result)

        # 3. Compute the calibrated magnitudes

        for result in photopolresults:

            if result.astrosource.is_calibrator:
                continue

            try:
                logger.debug(f"calibrating {result}")
                calibrate_photopolresult(result, photopolresults)
            except NoCalibratorsFound as e:
                logger.warning(f"no calibrators for {result}")
            except Exception as e:
                logger.error(f"I could not calibrate {result}: {e}.")

        # 5. Save the results

        for result in photopolresults:
            result.save()

        return photopolresults



class InstrumentHWP(ABC, Instrument):

    @property
    @abstractmethod
    def default_pol_method(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def rot_angles_required(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_instrumental_polarization(redf) -> InstrumentalPolarizationDict:
        raise NotImplementedError

    @classmethod
    def compute_relative_polarimetry(cls, polarimetry_group: 'PolarimetryGroup') -> List['PhotoPolResult']:
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

        # # if we want to disallow missing rotation angles

        # if len(polarimetry_group) != len(cls.rot_angles_required):
        #     raise Exception(f"Can not compute relative polarimetry for a group with {len(polarimetry_group)} reducedfits, it should be {len(cls.rot_angles_required)}.")

        # 1. Compute all aperture photometries

        common_apertures = cls.estimate_common_apertures(polarimetry_group, reductionmethod=REDUCTIONMETHODS.RELPHOT)

        r_ap = common_apertures.r_ap
        r_in = common_apertures.r_in
        r_out = common_apertures.r_out

        ap_fwhm = common_apertures.ap_fwhm

        fwhm_median = common_apertures.fwhm_stats.median

        with u.set_enabled_equivalencies(polarimetry_group[0].pixscale_equiv):
            aperpix = r_ap.to_value('pix')
            r_in_px = r_in.to_value('pix')
            r_out_px = r_out.to_value('pix')
            aperas = r_ap.to_value('arcsec')
            median_fwhm_as = fwhm_median.to_value('arcsec')           
        
        logger.debug(f"Computing aperture photometries for the {len(polarimetry_group)} reducedfits in the group (ap_fwhm = {ap_fwhm:.1f}, r_ap = {r_ap:.1f}, r_in = {r_in:.1f}, r_out = {r_out:.1f}).")

        aperphotresults = list(itertools.chain.from_iterable([
            cls.compute_aperture_photometry(redf, common_apertures)
            for redf in polarimetry_group
        ]))

        aperphotresult_pks = [aper.pk for aper in aperphotresults]

        # 2. Compute relative polarimetry for each source (uses the computed aperture photometries)

        logger.debug("Computing relative polarimetry.")

        photopolresults = list()

        for astrosource in group_sources:

            try:
                
                # if astrosource.calibrates.count() > 0:
                #     continue

                logger.debug(f"Computing relative polarimetry for {astrosource}.")

                aperphotresults = AperPhotResult.objects.filter(
                    pk__in=aperphotresult_pks,
                ).filter(
                    reducedfit__in = polarimetry_group,
                    astrosource = astrosource,
                    aperpix = aperpix,
                    r_in = r_in_px,
                    r_out = r_out_px,
                    flux_counts__isnull = False,
                )

                if len(aperphotresults) == 0:
                    logger.error(f"No aperphotresults found for {astrosource}")
                    continue

                # # if we want to disallow missing images:
                # if len(aperphotresults) != 2*len(cls.rot_angles_required):
                #     logger.error(f"There should be {2*len(cls.rot_angles_required)} aperphotresults for each astrosource in the group, there are {len(aperphotresults)} for {astrosource.name}.")
                #     continue

                values = get_column_values(aperphotresults, ['reducedfit__rotangle', 'flux_counts', 'flux_counts_err', 'pairs'])

                angles_L = list(sorted(set(values['reducedfit__rotangle'])))

                # # if we want to disallow missing rotator angles:
                # if len(angles_L) != len(cls.rot_angles_required):
                #     logger.warning(f"There should be {len(cls.rot_angles_required)} different angles, there are {len(angles_L)}.")

                fluxes = dict()
                flux_errors = dict()
                for pair, angle, flux, flux_err in zip(values['pairs'], values['reducedfit__rotangle'], values['flux_counts'], values['flux_counts_err']):
                    fluxes[(pair, angle)] = flux
                    flux_errors[(pair, angle)] = flux_err          

                theta = np.array([angle for angle in angles_L])

                FO = np.array([fluxes.get(('O', angle), np.nan) for angle in angles_L])
                dFO = np.array([flux_errors.get(('O', angle), np.nan) for angle in angles_L])

                FE = np.array([fluxes.get(('E', angle), np.nan) for angle in angles_L])
                dFE = np.array([flux_errors.get(('E', angle), np.nan) for angle in angles_L])

                logger.debug(f"{FO=}, {dFO=}, {FE=}, {dFE=}")

                # IOP4 astrocalibration atm works the other way, swap them here
                FO, dFO, FE, dFE = FE, dFE, FO, dFO

                # read possible special case from target source metadata
                only_pair = astrosource.metadata.get(f"{cls.name}.polarimetry.only_pair")

                if not only_pair:

                    logger.info(f"Computing stokes parameters with {cls.default_pol_method.name}")
                    
                    stokes_nocorr, fit_stats = cls.default_pol_method(theta, FO=FO, dFO=dFO, FE=FE, dFE=dFE)

                    logger.debug(f"{stokes_nocorr=}")
                    logger.debug(f"{fit_stats=}")
                    logger.info(f"{cls.default_pol_method.name} -> ({100*stokes_nocorr.p:.1f} +/ {100*stokes_nocorr.dp:.1f} %, {stokes_nocorr.chi:.1f} +/- {stokes_nocorr.dchi:.1f} º)")

                else:

                    logger.info(f"This source metadata indicates to use only the {only_pair} pair, computing stoke parameters with compute_stokes_HWP_fit_1pair")

                    if only_pair == 'O':
                        stokes_nocorr, fit_stats = compute_stokes_HWP_fit_1pair(theta, FO=FO, dFO=dFO)
                    elif only_pair == 'E':
                        stokes_nocorr, fit_stats = compute_stokes_HWP_fit_1pair(theta, FE=FE, dFE=dFE)

                logger.info(f"{astrosource.name} stokes_nocorr {stokes_nocorr} -> p = ({100*stokes_nocorr.p:.1f} +/ {100*stokes_nocorr.dp:.1f}) %, chi = ({stokes_nocorr.chi:.1f} +/- {stokes_nocorr.dchi:.1f}) º)")

                inst_pol_dict = cls.get_instrumental_polarization(reducedfit=polarimetry_group[0])

                stokes = stokes_nocorr.correct(**inst_pol_dict)

                logger.info(f"{astrosource.name} corrected stokes {stokes} -> p = ({100*stokes.p:.1f} +/ {100*stokes.dp:.1f}) %, chi = ({stokes.chi:.1f} +/- {stokes.dchi:.1f}) º)")

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

                # get median fwhm of the source
                _fwhms_as = list()
                for _redf in polarimetry_group:
                    for (_src, _pair), (_c, _fwhm) in common_apertures.centroids_and_fwhms[_redf].centroids_and_fwhms.items():
                        if _src == astrosource:
                            _fwhms_as.append(_fwhm.to_value('arcsec', equivalencies=_redf.pixscale_equiv))
                fwhm_source_as = np.nanmedian(_fwhms_as)

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
                    mag_inst = mag_inst,
                    mag_inst_err = mag_inst_err, 
                    mag_zp = mag_zp,
                    mag_zp_err = mag_zp_err,
                    # other info
                    flux_counts = flux,
                    _q_nocorr = stokes_nocorr.q,
                    _u_nocorr = stokes_nocorr.u,
                    _p_nocorr = stokes_nocorr.p,
                    _chi_nocorr = stokes_nocorr.chi,
                    # other info
                    aperpix = aperpix,
                    aperas = aperas,
                    fwhm = median_fwhm_as,
                    fwhm_source = fwhm_source_as,
                )

                result.aperphotresults.set(aperphotresults, clear=True)
                            
                photopolresults.append(result)

            except Exception as e:
                logger.error(f"{polarimetry_group}: relative polarimetry failed for {astrosource}: {e}")
                logger.debug(traceback.format_exc())
        
        # 3. Compute the calibrated magnitudes for non-calibrators in the group using the averaged zero point

        for result in photopolresults:

            if result.astrosource.is_calibrator:
                continue

            try:
                logger.debug(f"calibrating {result}")
                calibrate_photopolresult(result, photopolresults)
            except NoCalibratorsFound as e:
                logger.warning(f"no calibrators for {result}")
            except Exception as e:
                logger.error(f"I could not calibrate {result}: {e}.")

        # 4. Save results

        for result in photopolresults:
            result.save()

        return photopolresults
