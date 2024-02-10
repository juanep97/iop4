# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
import os
from pathlib import Path
import re
import astrometry
import numpy as np
import matplotlib as mplt
import matplotlib.patheffects
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder
from astropy.wcs import WCS
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from astropy.wcs.utils import fit_wcs_from_points
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
import itertools
import datetime
import math
import gc
import yaml
from importlib import resources

# iop4lib imports
from iop4lib.enums import IMGTYPES, BANDS, OBSMODES, SRCTYPES, INSTRUMENTS, REDUCTIONMETHODS
from .instrument import Instrument
from iop4lib.utils import imshow_w_sources, get_candidate_rank_by_matchs, get_angle_from_history, build_wcs_centered_on, get_simbad_sources
from iop4lib.utils.sourcedetection import get_sources_daofind, get_segmentation, get_cat_sources_from_segment_map, get_bkg
from iop4lib.utils.plotting import plot_preview_astrometry
from iop4lib.utils.astrometry import BuildWCSResult
from iop4lib.telescopes import OSNT090

# logging
import logging
logger = logging.getLogger(__name__)


import typing
if typing.TYPE_CHECKING:
    from iop4lib.db import RawFit, ReducedFit, Epoch

class DIPOL(Instrument):

    name = "DIPOL"
    telescope = OSNT090.name

    instrument_kw_L = ["ASI Camera (1)"]

    arcsec_per_pix = 0.134
    field_width_arcmin = 9.22
    field_height_arcmin = 6.28 
    
    gain_e_adu = 1

    required_masters = ['masterbias', 'masterflat', 'masterdark']


    # pre computed pairs distances to use in the astrometric calibrations
    # obtained from calibrated photometry fields
    
    disp_sign_mean, disp_sign_std = np.array([-2.09032765e+02,  1.65384209e-02]), np.array([4.13289109, 0.66159702])
    disp_mean, disp_std = np.abs(disp_sign_mean), disp_sign_std
    disp_std = np.array([15, 5])


    @classmethod
    def classify_juliandate_rawfit(cls, rawfit: 'RawFit'):
        """
        DIPOL files have JD keyword
        """
        import astropy.io.fits as fits
        jd = fits.getheader(rawfit.filepath, ext=0)["JD"]
        rawfit.juliandate = jd


    @classmethod
    def classify_imgtype_rawfit(cls, rawfit: 'RawFit'):
        """
        DIPOL files have IMAGETYP keyword: Light Frame (it can b, Bias Frame

        """
        from iop4lib.db.rawfit import RawFit
        import astropy.io.fits as fits

        with fits.open(rawfit.filepath) as hdul:
            if hdul[0].header['IMAGETYP'] == 'Bias Frame':
                rawfit.imgtype = IMGTYPES.BIAS
            elif hdul[0].header['IMAGETYP'] == 'Dark Frame':
                rawfit.imgtype = IMGTYPES.DARK
            elif hdul[0].header['IMAGETYP'] == 'Flat Field':
                rawfit.imgtype = IMGTYPES.FLAT
            elif hdul[0].header['IMAGETYP'] == 'Light Frame':
                if 'skyflat' in rawfit.header['OBJECT'].lower():
                    rawfit.imgtype = IMGTYPES.FLAT
                else:
                    rawfit.imgtype = IMGTYPES.LIGHT
            else:
                logger.error(f"Unknown image type for {rawfit.fileloc}.")
                rawfit.imgtype = IMGTYPES.ERROR
                raise ValueError

    @classmethod
    def classify_band_rawfit(cls, rawfit: 'RawFit'):
        """
            .. warning: 
                Red is in a differnt photometric system.
        """

        from iop4lib.db.rawfit import RawFit

        if 'FILTER' not in rawfit.header:
            if rawfit.imgtype == IMGTYPES.BIAS or rawfit.imgtype == IMGTYPES.DARK:
                rawfit.band = BANDS.NONE
            else:
                rawfit.band = BANDS.ERROR
                raise ValueError(f"Missing FILTER keyword for {rawfit.fileloc} which is not a bias or dark (it is a {rawfit.imgtype}).")
        elif rawfit.header['FILTER'] == "Red":  # TODO: they are not exacty red, they are in a different photometric system. Conversion must be implemented.
            rawfit.band = BANDS.R
        elif rawfit.header['FILTER'] == "Green":
            rawfit.band = BANDS.V
        elif rawfit.header['FILTER'] == "Blue":
            rawfit.band = BANDS.B
        else:
            rawfit.band = BANDS.ERROR
            raise ValueError(f"Unknown FILTER keyword for {rawfit.fileloc}: {rawfit.header['FILTER']}.")
    

    @classmethod
    def classify_obsmode_rawfit(cls, rawfit: 'RawFit'):
        """
        As of 2023-10-28, DIPOL polarimetry files have NOTES keyword with the angle like 'xxxx deg',
        photometry files have empty NOTES keyword.
        """

        if 'NOTES' in rawfit.header and not 'deg' in rawfit.header['NOTES']:
            rawfit.obsmode = OBSMODES.PHOTOMETRY
        else:
            rawfit.obsmode = OBSMODES.POLARIMETRY
            try:
                rawfit.rotangle = float(rawfit.header['NOTES'].split(' ')[0])
            except Exception as e:
                logger.error(f"Error parsing NOTES keyword for {rawfit.fileloc} as a float: {e}.")


    @classmethod
    def request_master(cls, rawfit, model, other_epochs=False):
        r""" Overriden Instrument associate_masters.
        
        DIPOL POLARIMETRY files are a cut of the full field, so when associating master calibration files, it needs to search a different size of images.
        For DIPOL PHOTOMETRY files, everything is the same as in the parent class.

        The full field images are 4144x2822, polarimetry images are 1100x900, the cut 
        position is saved in the images as 
            XORGSUBF 	0
            YORGSUBF 	0
        """

        if rawfit.obsmode == OBSMODES.PHOTOMETRY:
            return super().request_master(rawfit, model, other_epochs=other_epochs)
        
        # POLARIMETRY (see docstring)
        # everything should be the same as in the parent class except for the line changing args["imgsize"]

        from iop4lib.db import RawFit

        rf_vals = RawFit.objects.filter(id=rawfit.id).values().get()
        args = {k:rf_vals[k] for k in rf_vals if k in model.margs_kwL}
        
        args.pop("exptime", None) # exptime might be a building keywords (for flats and darks), but masters with different exptime can be applied
        args["epoch"] = rawfit.epoch # from .values() we only get epoch__id 

        args["imgsize"] = "4144x2822" # search for full field calibration frames

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
    def apply_masters(cls, reducedfit):
        """ Overriden for DIPOL (see DIPOL.request_master).
        
        The cut position is saved in the raw fit header as:
            XORGSUBF 	1500
            YORGSUBF 	1000
        """

        x_start = reducedfit.rawfit.header['XORGSUBF']
        y_start = reducedfit.rawfit.header['YORGSUBF']

        x_end = x_start + reducedfit.rawfit.header['NAXIS1']
        y_end = y_start + reducedfit.rawfit.header['NAXIS2']

        idx = np.s_[y_start:y_end, x_start:x_end]

        import astropy.io.fits as fits

        logger.debug(f"{reducedfit}: applying masters")

        rf_data = fits.getdata(reducedfit.rawfit.filepath)
        mb_data = fits.getdata(reducedfit.masterbias.filepath)[idx]

        if reducedfit.obsmode == OBSMODES.PHOTOMETRY:
            mf_data = fits.getdata(reducedfit.masterflat.filepath)[idx]
        else: # no flat fielding for polarimetry
            mf_data = 1.0

        if reducedfit.masterdark is not None:
            md_dark = fits.getdata(reducedfit.masterdark.filepath)[idx]
        else :
            logger.warning(f"{reducedfit}: no masterdark found, assuming dark current = 0, is this a CCD camera and it's cold?")
            md_dark = 0.0

        data_new = (rf_data - mb_data - md_dark*reducedfit.rawfit.exptime) / (mf_data)

        header_new = fits.Header()

        if not os.path.exists(os.path.dirname(reducedfit.filepath)):
            logger.debug(f"{reducedfit}: creating directory {os.path.dirname(reducedfit.filepath)}")
            os.makedirs(os.path.dirname(reducedfit.filepath))
        
        fits.writeto(reducedfit.filepath, data_new, header=header_new, overwrite=True)


    @classmethod
    def get_header_hintobject(self, rawfit):
        r""" Overriden for DIPOL, which are using the convention for the other_name field. 
        
        The regex used has been obtained from the notebook checking all keywords.
        """
        

        from iop4lib.db import AstroSource

        catalog = AstroSource.objects.exclude(srctype=SRCTYPES.CALIBRATOR).values('name', 'other_name')

        #pattern = re.compile(r"^([a-zA-Z0-9]{4,}|[a-zA-Z0-9]{1,3}(_[a-zA-Z0-9]+)?)(?=_|$)")
        pattern = re.compile(r"^([a-zA-Z0-9]{1,3}_[a-zA-Z0-9]+|[a-zA-Z0-9]{4,})(?=_|$)")
        
        obj_kw = rawfit.header['OBJECT']
        
        match = pattern.match(obj_kw)

        def get_invariable_str(s):
            return s.replace(' ', '').replace('-','').replace('+','').replace('_','').upper()

        if match:
            
            search_str = match.group(0)
            
            for source in catalog:
                if not source['other_name']:
                    continue
                if get_invariable_str(search_str) in get_invariable_str(source['other_name']):
                    return AstroSource.objects.get(name=source['name'])

            for source in catalog:
                if get_invariable_str(search_str) in get_invariable_str(source['name']):
                    return AstroSource.objects.get(name=source['name'])
                
        return None
      
    
    @classmethod
    def get_header_hintcoord(cls, rawfit):
        """ Overriden for DIPOL

        As of 2023-10-23, DIPOL does not inclide RA and DEC in the header, RA and DEC will be derived from the object name.
        """     
        
        return rawfit.header_hintobject.coord            
            
        
    @classmethod
    def get_astrometry_size_hint(cls, rawfit: 'RawFit'):
        """ Implement Instrument.get_astrometry_size_hint for DIPOL.

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
    
    @classmethod
    def get_astrometry_position_hint(cls, rawfit: 'RawFit', allsky=False, n_field_width=1.5, hintsep=None):
        """ Get the position hint from the FITS header as an astrometry.PositionHint.
        
        Parameters
        ----------
        allsky: bool, optional
            If True, the hint will cover the whole sky, and n_field_width and hintsep will be ignored.
        n_field_width: float, optional
            The search radius in units of field width. Default is 1.5.
        hintsep: Quantity, optional
            The search radius in units of degrees.
        """        

        hintcoord = cls.get_header_hintcoord(rawfit)

        if rawfit.header["XBINNING"] != 2:
            logger.error(f"Cannot compute astrometry for {rawfit} because of the binning: {rawfit.header['XBINNING']}.")
            return None
        
        if allsky:
            hintsep = 180.0 * u.deg
        else:
            if hintsep is None:
                hintsep = (n_field_width * cls.field_width_arcmin*u.Unit("arcmin"))

        return astrometry.PositionHint(ra_deg=hintcoord.ra.deg, dec_deg=hintcoord.dec.deg, radius_deg=hintsep.to_value(u.deg))
    
    @classmethod
    def has_pairs(cls, fit_instance: 'ReducedFit' or 'RawFit') -> bool:
        """ DIPOL ALWAYS HAS PAIRS """
        return True





    @classmethod
    def _estimate_positions_from_segments(cls, redf=None, data=None, fwhm=None, npixels=64, n_seg_threshold=3.0, centering=2/3):

        if fwhm is None:
            if redf is not None and redf.header_hintobject.srctype == SRCTYPES.STAR and redf.exptime <= 5:
                # the star might be too bright, we need to smooth the image not to get too fake sources
                fwhm = 80.0
            else:
                fwhm = 1.0

        # get the sources positions

        if data is None:
            data = redf.mdata
 
        height, width = data.shape

        bkg = get_bkg(data, filter_size=5, box_size=width//10)
        imgdata_bkg_substracted = data - bkg.background
        seg_threshold = n_seg_threshold * bkg.background_rms
        
        segment_map, convolved_data = get_segmentation(imgdata_bkg_substracted, fwhm=fwhm, npixels=npixels, threshold=seg_threshold)
        if segment_map is None:
            return list()
        else:
            seg_cat, positions, tb = get_cat_sources_from_segment_map(segment_map, imgdata_bkg_substracted, convolved_data)
        
        if centering:
            # select only the sources in the center
            cx, cy = width//2, height//2
            idx = np.abs(positions[:,0]-cx) < centering * width / 2
            idx = idx & (np.abs(positions[:,1]-cy) < centering * height / 2)
            positions = positions[idx]

        return positions

    @classmethod
    def build_wcs(cls, reducedfit: 'ReducedFit', summary_kwargs : dict = None, method=None):
        """ Overriden Instrument build_wcs.
        
        While for PHOTOMETRY observations, DIPOL has a wide field which can be astrometrically calibrated, 
        POLARIMETRY files are small with only the source field ordinary and extraordianty images in the center (to save up space).
        In some ocassions, there might be some close source also in the field.

        Therefore, to calibrate polarimetry files, we just give it a WCS centered on the source.

        For PHOTOMETRY files, we use the parent class method, but we set some custom shotgun_params_kwargs to account
        for the low flux and big size of the images.

        """

        if summary_kwargs is None:
            summary_kwargs = {'build_summary_images':True, 'with_simbad':True}

        if method is not None:
            logger.warning(f"Calling {method} for {reducedfit}.")
            return method(reducedfit, summary_kwargs=summary_kwargs)

        from iop4lib.db import ReducedFit, AstroSource

        target_src = reducedfit.header_hintobject

        if reducedfit.obsmode == OBSMODES.PHOTOMETRY:
            return super().build_wcs(reducedfit, shotgun_params_kwargs=cls._build_shotgun_params(reducedfit), summary_kwargs=summary_kwargs)
        elif reducedfit.obsmode == OBSMODES.POLARIMETRY:

            # Gather some info to perform a good decision on which methods to use

            n_estimate = len(cls._estimate_positions_from_segments(redf=reducedfit, n_seg_threshold=1.3, npixels=64, centering=None))
            n_estimate_centered = len(cls._estimate_positions_from_segments(redf=reducedfit, n_seg_threshold=1.3, npixels=64, centering=2/3))
            redf_phot = ReducedFit.objects.filter(instrument=reducedfit.instrument,
                                                  sources_in_field__in=[reducedfit.header_hintobject], 
                                                  obsmode=OBSMODES.PHOTOMETRY, 
                                                  flags__has=ReducedFit.FLAGS.BUILT_REDUCED).first()
            n_expected_simbad_sources = len(get_simbad_sources(reducedfit.header_hintobject.coord, radius=(reducedfit.width*cls.arcsec_per_pix*u.arcsec)))
            n_expected_calibrators = AstroSource.objects.filter(calibrates__in=[reducedfit.header_hintobject]).count()

            # log the variables above

            logger.debug(f"{target_src.srctype=}")
            logger.debug(f"{n_estimate=}")
            logger.debug(f"{n_estimate_centered=}")
            logger.debug(f"{redf_phot=}")
            logger.debug(f"{n_expected_simbad_sources=}")
            logger.debug(f"{n_expected_calibrators=}")
            
            with open(resources.files("iop4lib.instruments") / "dipol_astrometry.yaml") as f:
                dipol_astrometry = yaml.safe_load(f)

            def apply_comparison(val1, val2, op):
                if op == "eq":
                    return val1 == val2
                elif op == "ne":
                    return val1 != val2
                elif op == "gt":
                    return val1 > val2
                elif op == "gte":
                    return val1 >= val2
                elif op == "lt":
                    return val1 < val2
                elif op == "lte":
                    return val1 <= val2
                elif op is None or op == "":
                    return val1 == val2

            def check_conditions(branch, context):
                if 'conds' in branch and len(branch['conds']) > 0:
                    keys, ops, vals = zip(*[(*k.split('__'),v) if '__' in k else (k,None,v) for k,v in branch['conds'].items()])
                    satisfies_conds = all([apply_comparison(context[k], v, op) for k,op,v in zip(keys, ops, vals)])
                else:
                    satisfies_conds = True
                return satisfies_conds

            def find_attempts(rules, context):
                for rule in rules:
                    if check_conditions(rule, context):
                        if 'attempts' in rule:
                            # get the attempts as a list, fetching the definitions if necessary
                            attempts = [dipol_astrometry['attempt_defs'][attempt] if isinstance(attempt, str) else attempt for attempt in rule['attempts']]
                            # filter the attempts by the conditions
                            return list(filter(lambda attempt: check_conditions(attempt, context), attempts))
                        else:
                            return find_attempts(rule['rules'], context)
                return None
        
            attempts = find_attempts(dipol_astrometry['rules'], context = {
                'n_estimate': n_estimate,
                'n_estimate_centered': n_estimate_centered,
                'n_expected_simbad_sources': n_expected_simbad_sources,
                'n_expected_calibrators': n_expected_calibrators,
                'exptime': reducedfit.exptime,
                'srctype': target_src.srctype,
                'srcname': target_src.name,
                'redf_phot': redf_phot,
            })

            
            for attempt in attempts:
                logger.debug(f"Trying attempt: {attempt} for {reducedfit}.")

                m = getattr(cls, attempt['method'])
                args = attempt['args']

                args_keys = list(args.keys())
                args_values_list = list(itertools.product(*args.values()))
                args_dict_list = [dict(zip(args_keys, args_vals)) for args_vals in args_values_list]

                build_wcs = False

                for args_dict in args_dict_list:
                    logger.info(f"Attempt: {attempt}: {m.__name__} with {args_dict} for {reducedfit}.")
                    if (build_wcs := m(reducedfit, summary_kwargs=summary_kwargs, **args_dict)):
                        break
                
                build_wcs.info["attempt"] = attempt
                build_wcs.info["m.__name__"] = m.__name__
                build_wcs.info["args"] = args_dict
                build_wcs.info["n_estimate"] = n_estimate
                build_wcs.info["n_estimate_centered"] = n_estimate_centered
                build_wcs.info["n_expected_simbad_sources"] = n_expected_simbad_sources
                build_wcs.info["n_expected_calibrators"] = n_expected_calibrators

                if build_wcs:
                    break

            return build_wcs
        
        else:
            logger.error(f"Unknown obsmode {reducedfit.obsmode} for {reducedfit}.")
            raise ValueError
            
        
    @classmethod
    def _build_shotgun_params(cls, redf: 'ReducedFit'):
        shotgun_params_kwargs = dict()

        shotgun_params_kwargs["keep_n_seg"] = [300]
        shotgun_params_kwargs["border_margin_px"] = [20]
        shotgun_params_kwargs["output_logodds_threshold"] = [14]
        shotgun_params_kwargs["n_rms_seg"] = [1.5, 1.2, 1.0]
        shotgun_params_kwargs["bkg_filter_size"] = [11] 
        shotgun_params_kwargs["bkg_box_size"] = [32]
        shotgun_params_kwargs["seg_fwhm"] = [1.0]
        shotgun_params_kwargs["npixels"] = [32, 8]
        shotgun_params_kwargs["seg_kernel_size"] = [None]
        shotgun_params_kwargs["allsky"] = [False]

        shotgun_params_kwargs["d_eps"] = [4.0]
        shotgun_params_kwargs["dx_eps"] = [4.0]
        shotgun_params_kwargs["dy_eps"] = [2.0]
        shotgun_params_kwargs["dx_min"] = [150]
        shotgun_params_kwargs["dx_max"] = [300]
        shotgun_params_kwargs["dy_min"] = [0]
        shotgun_params_kwargs["dy_max"] = [50]
        shotgun_params_kwargs["d_min"] = [150]
        shotgun_params_kwargs["d_max"] = [250]
        shotgun_params_kwargs["bins"] = [400]
        shotgun_params_kwargs["hist_range"] = [(0,500)]

        shotgun_params_kwargs["position_hint"] = [redf.get_astrometry_position_hint(allsky=False)]
        shotgun_params_kwargs["size_hint"] = [redf.get_astrometry_size_hint()]

        return shotgun_params_kwargs


    @classmethod
    def _build_wcs_for_polarimetry_images_photo_quads(cls, redf: 'ReducedFit', summary_kwargs : dict = None, n_seg_threshold=1.5, npixels=32, min_quad_distance=4.0, fwhm=None, centering=None):

        if summary_kwargs is None:
            summary_kwargs = {'build_summary_images':True, 'with_simbad':True}

        from iop4lib.db import ReducedFit

        if (target_src := redf.header_hintobject) is None:
            raise Exception("No target source found in header, cannot build WCS.")

        # the polarimetry field

        redf_pol = redf

        # select an already solved photometry field

        redf_phot = ReducedFit.objects.filter(instrument=redf_pol.instrument, 
                                              sources_in_field__in=[target_src], 
                                              obsmode=OBSMODES.PHOTOMETRY, 
                                              flags__has=ReducedFit.FLAGS.BUILT_REDUCED).first()
        
        if redf_phot is None:
            logger.error(f"No astro-calibrated photometry field found for {redf_pol}.")
            return BuildWCSResult(success=False)

        logger.debug(f"Invoked with {n_seg_threshold=}, {npixels=}")

        # get the subframe of the photometry field that corresponds to this polarimetry field, (approx)
        x_start = redf_pol.rawfit.header['XORGSUBF']
        y_start = redf_pol.rawfit.header['YORGSUBF']

        x_end = x_start + redf_pol.rawfit.header['NAXIS1']
        y_end = y_start + redf_pol.rawfit.header['NAXIS2']

        idx = np.s_[y_start:y_end, x_start:x_end]

        photdata_subframe = redf_phot.mdata[idx] # if we use the hash_ish_old, which is not invariant under fliiping, we need to flip the image in y (redf_phot.mdata[idx][::-1,:])

        # find 10 brightest sources in each field

        sets_L = list()

        for redf, data in zip([redf_pol, redf_phot], [redf_pol.mdata, photdata_subframe]):

            positions = cls._estimate_positions_from_segments(redf=redf, data=data, n_seg_threshold=n_seg_threshold, npixels=npixels, centering=centering, fwhm=fwhm)
            positions = positions[:10]

            sets_L.append(positions)
        
        logger.debug(f"Using {len(sets_L[0])} sources in polarimetry field and {len(sets_L[1])} in photometry field.")

        if summary_kwargs['build_summary_images']:
            logger.debug(f"Building summary image for astrometry detected sources.")

            fig = mplt.figure.Figure(figsize=(12,6), dpi=iop4conf.mplt_default_dpi)
            axs = fig.subplots(nrows=1, ncols=2)

            for ax, data, positions in zip(axs, [redf_pol.mdata, photdata_subframe], sets_L):
                imshow_w_sources(data, pos1=positions, ax=ax)
                candidates_aps = CircularAperture(positions[:2], r=10.0)
                candidates_aps.plot(ax, color="b")
                for i, (x,y) in enumerate(positions):
                    ax.text(x, y, f"{i}", color="orange", fontdict={"size":14, "weight":"bold"})#, verticalalignment="center", horizontalalignment="center") 
                ax.plot([data.shape[1]//2], [data.shape[0]//2], '+', color='y', markersize=10)
                
            axs[0].set_title("Polarimetry field")
            axs[1].set_title("Photometry field")
            fig.savefig(Path(redf_pol.filedpropdir) / "astrometry_detected_sources.png", bbox_inches="tight")
            fig.clf()

        # Build the quads for each field
        quads_1 = np.array(list(itertools.combinations(sets_L[0], 4)))
        quads_2 = np.array(list(itertools.combinations(sets_L[1], 4)))

        if len(quads_1) == 0 or len(quads_2) == 0:
            logger.error(f"No quads found in {redf_pol} and {redf_phot}, returning success = False.")
            return BuildWCSResult(success=False)
        
        from iop4lib.utils.quadmatching import hash_ish, distance, order, qorder_ish, find_linear_transformation
        hash_func, qorder = hash_ish, qorder_ish

        hashes_1 = np.array([hash_func(quad) for quad in quads_1])
        hashes_2 = np.array([hash_func(quad) for quad in quads_2])

        all_indices = np.array(list(itertools.product(range(len(quads_1)),range(len(quads_2)))))
        all_distances = np.array([distance(hashes_1[i], hashes_2[j]) for i,j in all_indices])

        idx = np.argsort(all_distances)
        all_indices = all_indices[idx]
        all_distances = all_distances[idx]

        # selected indices some nice indices

        #best_i, best_j = all_indices[0]

        # if (min_distance_found := distance(hash_func(quads_1[best_i]),hash_func(quads_2[best_j]))) > 4.0: # corresponds to more than 1px error per point in the quad
        #     logger.error(f"Best quads i,j=[{best_i},{best_j}] matched has distance {min_distance_found:.3f} > 4.0, returning success = False.")
        #     return BuildWCSResult(success=False, wcslist=None, info={'redf_phot__pk':redf_phot.pk, 'redf_phot__fileloc':redf_phot.fileloc})

        # better this method:

        # the best 5 that have less than 1px of error per quad (4 points)
 
        idx_selected = np.where(all_distances < min_quad_distance)[0] 
        indices_selected = all_indices[idx_selected]
        distances_selected = all_distances[idx_selected]
        
        if len(idx_selected) == 0:
            logger.error(f"No quads with distance < {min_quad_distance}, minimum at {min(all_distances)=} returning success = False.")
            return BuildWCSResult(success=False, wcslist=None, info={'redf_phot__pk':redf_phot.pk, 'redf_phot__fileloc':redf_phot.fileloc}) 
        else:
            idx_selected = np.argsort(distances_selected)[:5]
            indices_selected = all_indices[idx_selected]
            distances_selected = all_distances[idx_selected]

        # get linear transforms
        logger.debug(f"Selected {len(indices_selected)} quads with distance < {min_quad_distance}. I will get the one with less deviation from the median linear transform.")

        R_L, t_L = zip(*[find_linear_transformation(qorder(quads_1[i]), qorder(quads_2[j])) for i,j in indices_selected])
        logger.debug(f"{t_L=}")
        

        logger.debug(f"Filtering out big translations (<1020px)")

        _indices_selected = indices_selected[np.array([np.linalg.norm(t) < 1020 for t in t_L])]

        logger.debug(f"Filtered to {len(_indices_selected)} quads with distance < 4.0 and translation < 1020px.")

        if len(_indices_selected) == 0:
            logger.error(f"No quads with distance < {min_quad_distance} and translation < 1000px, building summary image of the 3 best quads and returning success = False.")

            colors = [color for color in mplt.rcParams["axes.prop_cycle"].by_key()["color"]]

            fig = mplt.figure.Figure(figsize=(12,6), dpi=iop4conf.mplt_default_dpi)
            axs = fig.subplots(nrows=1, ncols=2)

            for (i, j), color in list(zip(indices_selected, colors))[:3]: 

                tij = find_linear_transformation(qorder(quads_1[i]), qorder(quads_2[j]))[1]

                for ax, data, quad, positions in zip(axs, [redf_pol.mdata, photdata_subframe], [quads_1[i], quads_2[j]], sets_L):
                    imshow_w_sources(data, pos1=positions, ax=ax)
                    x, y = np.array(order(quad)).T
                    ax.fill(x, y, edgecolor='k', fill=True, facecolor=mplt.colors.to_rgba(color, alpha=0.2))
                    for pi, p in enumerate(np.array((qorder(quad)))):
                        xp = p[0]
                        yp = p[1]
                        ax.text(xp, yp, f"{pi}", fontsize=16, color=color, path_effects=[mplt.patheffects.Stroke(linewidth=1, foreground='black'), mplt.patheffects.Normal()])

                fig.suptitle(f"dist({i},{j})={distance(hash_func(quads_1[i]),hash_func(quads_2[j])):.3f}, norm(t) = {np.linalg.norm(tij):.0f} px", y=0.83)

            axs[0].set_title("Polarimetry")
            axs[1].set_title("Photometry")
            
            fig.savefig(Path(redf_pol.filedpropdir) / "astrometry_matched_quads.png", bbox_inches="tight")
            fig.clf()

            return BuildWCSResult(success=False, wcslist=None, info={'redf_phot__pk':redf_phot.pk, 'redf_phot__fileloc':redf_phot.fileloc})
        else:
            indices_selected = _indices_selected
        
        R_L, t_L = zip(*[find_linear_transformation(qorder(quads_1[i]), qorder(quads_2[j])) for i,j in indices_selected])


        # get the closest one to the t_L mean
        median_t = np.median(t_L, axis=0)

        logger.debug(f"{median_t=}")

        delta_t = np.array([np.linalg.norm(t - median_t) for t in t_L])
        indices_selected = indices_selected[np.argsort(delta_t)]
        best_i, best_j = indices_selected[0]
        
        logger.debug(f"Selected the quads [{best_i},{best_j}]")

        logger.debug(f"t = {t_L[np.argmin(delta_t)]}")

        # build_summary_images, replace indices_selected by all_indices if no R,t filtering was done.
        if summary_kwargs['build_summary_images']:
            logger.debug(f"Building summary image for quad matching.")

            colors = [color for color in mplt.rcParams["axes.prop_cycle"].by_key()["color"]]

            fig = mplt.figure.Figure(figsize=(12,6), dpi=iop4conf.mplt_default_dpi)
            axs = fig.subplots(nrows=1, ncols=2)

            for (i, j), color in list(zip(indices_selected, colors))[:1]: 
                
                tij = find_linear_transformation(qorder(quads_1[i]), qorder(quads_2[j]))[1]

                for ax, data, quad, positions in zip(axs, [redf_pol.mdata, photdata_subframe], [quads_1[i], quads_2[j]], sets_L):
                    imshow_w_sources(data, pos1=positions, ax=ax)
                    x, y = np.array(order(quad)).T
                    ax.fill(x, y, edgecolor='k', fill=True, facecolor=mplt.colors.to_rgba(color, alpha=0.2))
                    for pi, p in enumerate(np.array((qorder(quad)))):
                        xp = p[0]
                        yp = p[1]
                        ax.text(xp, yp, f"{pi}", fontsize=16, color=color, path_effects=[mplt.patheffects.Stroke(linewidth=1, foreground='black'), mplt.patheffects.Normal()])

                fig.suptitle(f"dist({i},{j})={distance(hash_func(quads_1[i]),hash_func(quads_2[j])):.3f}, norm(t) = {np.linalg.norm(tij):.0f} px", y=0.83)

            axs[0].set_title("Polarimetry")
            axs[1].set_title("Photometry")
            
            fig.savefig(Path(redf_pol.filedpropdir) / "astrometry_matched_quads.png", bbox_inches="tight")
            fig.clf()

        # Build the WCS

        # give an unique ordering to the quads

        quads_1 = [qorder_ish(quad) for quad in quads_1]
        quads_2 = [qorder_ish(quad) for quad in quads_2]

        # get the pre wcs with the target in the center of the image

        angle_mean, angle_std = get_angle_from_history(redf_pol, target_src)
        if 'FLIPSTAT' in redf_pol.rawfit.header: 
            # TODO: check that indeed the mere presence of this keyword means that the image is flipped, without the need of checking the value. 
            # FLIPSTAT is a MaximDL thing only, but it seems that the iamge is flipped whenever the keyword is present, regardless of the value.
            angle = - angle_mean
        else:
            angle = angle_mean

        logger.debug(f"Using {angle=} for pre wcs.")

        pre_wcs = build_wcs_centered_on((redf_pol.width//2,redf_pol.height//2), redf=redf_phot, angle=angle)
        
        # get the pixel arrays in the polarimetry field and in the FULL photometry field to relate them
        pix_array_1 = np.array(list(zip(*[(x,y) for x,y in quads_1[best_i]])))
        pix_array_2 = np.array(list(zip(*[(x+x_start,y+y_start) for x,y in quads_2[best_j]])))

        # fit the WCS so the pixel arrays in 1 correspond to the ra/dec of the pixel array in 2
        wcs1 = fit_wcs_from_points(pix_array_1,  redf_phot.wcs1.pixel_to_world(*pix_array_2), projection=pre_wcs)
        wcs2 = fit_wcs_from_points(pix_array_1,  redf_phot.wcs2.pixel_to_world(*pix_array_2), projection=pre_wcs)

        wcslist = [wcs1, wcs2]

        if summary_kwargs['build_summary_images']:
            logger.debug(f"Building summary image for astrometry.")
            fig = mplt.figure.Figure(figsize=(6,6), dpi=iop4conf.mplt_default_dpi)
            ax = fig.subplots(nrows=1, ncols=1, subplot_kw={'projection': wcslist[0]})
            plot_preview_astrometry(redf_pol, with_simbad=True, has_pairs=True, wcs1=wcslist[0], wcs2=wcslist[1], ax=ax, fig=fig) 
            fig.savefig(Path(redf_pol.filedpropdir) / "astrometry_summary.png", bbox_inches="tight")
            fig.clear()            


        result = BuildWCSResult(success=True, wcslist=wcslist, info={'redf_phot__pk':redf_phot.pk, 'redf_phot__fileloc':redf_phot.fileloc, n_seg_threshold:n_seg_threshold, npixels:npixels})

        return result



    @classmethod
    def _build_wcs_for_polarimetry_images_catalog_matching(cls, redf: 'ReducedFit', summary_kwargs : dict = None, n_seg_threshold=1.5, npixels=64, fwhm=None, centering=2/3):
        r""" Build WCS for DIPOL polarimetry images by matching the found sources positions with the catalog.

        .. warning::
            When there are some DB sources in the field it probably works, but when there are not, it might give a result that is centered on the wrong source!
        
            It is safer to use the method _build_wcs_for_polarimetry_images_photo_quads, which uses the photometry field and quad matching to build the WCS.
            
        """

        if summary_kwargs is None:
            summary_kwargs = {'build_summary_images':True, 'with_simbad':True}

        # disp_allowed_err = 1.5*cls.disp_std 
        disp_allowed_err =  np.array([30,30]) # most times should be much smaller (1.5*std)
        # but in bad cases, this is ~1 sigma of the gaussians

        logger.debug(f"{redf}: building WCS for DIPOL polarimetry images.")
        
        from iop4lib.db import AstroSource
        
        # define target astro source
        target_src = redf.header_hintobject

        if target_src is None:
            raise Exception("No target source found in header, cannot build WCS.")

        data = redf.mdata
        cx, cy = redf.width//2, redf.height//2

        positions = cls._estimate_positions_from_segments(redf=redf, n_seg_threshold=n_seg_threshold, npixels=npixels, fwhm=fwhm, centering=centering)

        if len(positions) == 0:
            logger.error(f"{redf}: Found no sources in the field, cannot build WCS.")
            return BuildWCSResult(success=False)
        else:
            logger.debug(f"{redf}: Found {len(positions)} with {n_seg_threshold=} {npixels=} sources in the field.")
        
        # if the are more than 100, work with only 20 brightest (they are already sorted by flux)

        if len(positions) > 20:
            positions = positions[:20]

        if summary_kwargs['build_summary_images']:
            # plot summary of detected sources
            fig = mplt.figure.Figure(figsize=(6,6), dpi=iop4conf.mplt_default_dpi)
            ax = fig.subplots(nrows=1, ncols=1)
            imshow_w_sources(data, pos1=positions, ax=ax)
            candidates_aps = CircularAperture(positions[:2], r=10.0)
            candidates_aps.plot(ax, color="b")
            for i, (x,y) in enumerate(positions[:15]):
                ax.text(x, y, f"{i}", color="orange", fontdict={"size":14, "weight":"bold"})#, verticalalignment="center", horizontalalignment="center") 
            ax.plot([cx], [cy], '+', color='y', markersize=10)
            if centering:
                ax.axhline(cy-centering*redf.height/2, xmin=0, xmax=redf.width, color='y', linestyle='--')
                ax.axhline(cy+centering*redf.height/2, xmin=0, xmax=redf.width, color='y', linestyle='--')
                ax.axvline(cx-centering*redf.width/2, ymin=0, ymax=redf.height, color='y', linestyle='--')
                ax.axvline(cx+centering*redf.width/2, ymin=0, ymax=redf.height, color='y', linestyle='--')
            ax.set_title(f"Detected sources {npixels=}, {n_seg_threshold=}")
            fig.savefig(Path(redf.filedpropdir) / "astrometry_detected_sources.png", bbox_inches="tight")
            fig.clear()

        angle_mean, angle_std = get_angle_from_history(redf, target_src)

        # DIPOL polarimery images seem to be flipped vertically, which results in negative angle
        # TODO: watch this FLIP thing, check that indeed this is the behaviour
        if 'FLIPSTAT' in redf.rawfit.header:
            angle = - angle_mean
        else:
            angle = angle_mean
        
        logger.debug(f"Angle {angle=}")

        # Now, if there is only two sources, they must be the ordinary and extraordinary images. We 
        # use them, if they are not, the procedure failed, raise exception.

        # Otherwise, they could correspond to the calibrators or other sources in the field.
        # We can use catalog sources in the same field (calibrators) to ascertain which of the candidates are 
        # the ordinary and extraordinary images, by matching the positions of sources in the field. 
        # This requires at least one calibrator in the field. It can be checked with a initial approximated 
        # WCS (pre_wcs below). If no calibrators in the DB is found we can try querying simbad, to see if 
        # there is some known star close, and them use them.

        pre_wcs = build_wcs_centered_on((cx,cy), redf=redf, angle=angle)

        # get list if calibrators for this source in the DB expected to be inside the subframe
        expected_sources_in_field = AstroSource.get_sources_in_field(
            wcs=pre_wcs, 
            width=redf.width, 
            height=redf.height,
            qs=AstroSource.objects.filter(calibrates__in=[target_src]).all())

        if len(positions) == 2:
            logger.debug("Found only two sources in the field, assuming they are the ordinary and extraordinary images.")
            target_O, target_E = positions
        else:

            if len(expected_sources_in_field) == 0:
                logger.warning(f"{redf}: No other DB sources in the field to check. Checking SIMBAD sources...")
                simbad_search_radius = Angle(cls.arcsec_per_pix*redf.width/3600, unit="deg")
                expected_sources_in_field = get_simbad_sources(target_src.coord, simbad_search_radius, 10, exclude_self=True)
                expected_sources_in_field = [src for src in expected_sources_in_field if src.is_in_field(pre_wcs, redf.height, redf.width)]

            if len(expected_sources_in_field) > 0:
                # The function get_candidate_rank_by_matchs returns a rank for each candidate, the higher the rank, 
                # the more likely it is to be the target source according to the matches with the catalog sources.

                # filter the positionsto rank only to those that could be pairs
                from iop4lib.utils.quadmatching import distance
                from iop4lib.utils.sourcepairing import get_best_pairs

                list1, list2, _, _  = get_best_pairs(*zip(*itertools.product(positions,positions)), cls.disp_sign_mean, disp_sign_err=disp_allowed_err)

                positions_to_rank = list()
                positions_to_rank.extend(list1)
                positions_to_rank.extend(list2)
                positions_to_rank = np.array(positions_to_rank)
                logger.debug(f"{positions_to_rank=}")

                if len(positions_to_rank) < 2:
                    logger.error("No pairs found, returning success = False.")
                    return BuildWCSResult(success=False)

                # N_max_to_rank = 10
                # positions_to_rank = positions_to_rank[:N_max_to_rank]

                ranks, _ = zip(*[get_candidate_rank_by_matchs(redf, pos, angle=angle, r_search=15, calibrators=expected_sources_in_field) for pos in positions_to_rank])
                ranks = np.array(ranks)

                # idx_sorted_by_rank = np.argsort(ranks)[::-1] # non stable sort
                idx_sort_by_rank = np.argsort(-ranks, kind="stable") # stable sort

                with np.printoptions(precision=2, suppress=True):
                    logger.debug(f"{positions_to_rank=}")
                    logger.debug(f"{ranks=}")
                    logger.debug(f"{idx_sort_by_rank=}")

                # If the procedure worked, the first two sources should be the 
                # ordinary and extraordinary images, which should have the most similar
                # fluxes, there fore check if they are next to each others.
                # if they are not, the procedure might have failed, give a warning.

                if abs(idx_sort_by_rank[0] - idx_sort_by_rank[1]) != 1:
                    logger.warning("adyacent by rank flux mismatch detected")

                if not any([(np.isfinite(r) and r>0) for r in ranks]):
                    logger.error("None of the ranks worked, returning success = False.")
                    return BuildWCSResult(success=False)
                else:
                    logger.debug(f"Ranks discriminated well")
                    pre_list1, pre_list2 = zip(*itertools.product([positions_to_rank[i] for i in range(len(positions_to_rank)) if np.isfinite(ranks[i]) and ranks[i] >= np.nanmax(ranks)], positions))

                # log some debug info about the pairs diference and the difference with respect the expected disp_sign_mean

                for i, (pos1, pos2) in enumerate(zip(pre_list1, pre_list2)):
                    dist = distance(pos1, pos2)
                    disp = np.abs(np.subtract(pos1, pos2))
                    diff = np.abs(np.subtract(pos1, pos2))-np.abs(cls.disp_sign_mean)
                    with np.printoptions(precision=1, suppress=True):
                        logger.debug(f"{i=},\t{pos1=!s},\t{pos2=!s},\t{dist=:.2f},\t{disp=!s},\t{diff=!s}")

                # get the best pairs according to the disp_sign_mean 
                # since we dont know if pre_list1 is the ordinary or extraordinary image, try with 
                # disp_sign_mean and -disp_sign_mean

                list1, list2, d0_new, disp_sign_new = get_best_pairs(pre_list1, pre_list2, cls.disp_sign_mean, disp_sign_err=disp_allowed_err)

                with np.printoptions(precision=1, suppress=True):
                    logger.debug(f"{list1=}, {list2=}, {d0_new=}, {disp_sign_new=}")

                if len(list1) == 0:
                    list1, list2, d0_new, disp_sign_new = get_best_pairs(pre_list1, pre_list2, -cls.disp_sign_mean, disp_sign_err=disp_allowed_err)

                    with np.printoptions(precision=1, suppress=True):
                        logger.debug(f"{list1=}, {list2=}, {d0_new=}, {disp_sign_new=}")

                    if len(list1) == 0:
                        logger.error("No pairs found, returning success = False.")
                        return BuildWCSResult(success=False)
                
                target_O, target_E = list1[0], list2[0]
                
            else:
                
                logger.error("No SIMBAD sources in the field to check either.")

                return BuildWCSResult(success=False)

        # Make the ordinary image the one in the right always.

        if target_O[0] < target_E[0]:
            target_O, target_E = target_E, target_O

        # from preliminary astrometry of photometry images
        # if they are not pairs, the procedure definitely failed, raise exception

        if not np.isclose(np.abs(target_E[0]-target_O[0]), cls.disp_mean[0], atol=disp_allowed_err[0]):
            logger.error(f"These are not pairs, x mismatch detected according to hard-coded pair distance:  disp x = {np.abs(target_E[0]-target_O[0]):.0f} px)")
            return BuildWCSResult(success=False)
            
        if not np.isclose(np.abs(target_E[1]-target_O[1]), cls.disp_mean[1], atol=disp_allowed_err[1]):
            logger.error(f"These are not pairs, y mismatch detected according to hard-coded pair distance: disp y = {np.abs(target_E[1]-target_O[1]):.0f} px")
            return BuildWCSResult(success=False)

        # WCS for Ordinary and Extraordinary images

        # Now that we have positions of the ordinary and extraordinary images, we can the WCS for each set of pairs.
        # We could always use the pre-computed angles but there might be small variations -not so small- near the border.
        # If there are enough calibrators, we might use their positions to fit a WCS object, which should be more precise.
        # If there are not enough calibrators, we can use the pre-computed angles.

        calibrators_in_field = [src for src in AstroSource.objects.filter(calibrates__in=[target_src]).all() if src.is_in_field(pre_wcs, redf.height, redf.width)]

        logger.debug(f"Found {len(calibrators_in_field)} calibrators in field for {target_src.name}: {calibrators_in_field}")

        if len(calibrators_in_field) <= 1 or len(positions) <= 2:
            logger.warning(f"Using pre-computed angle {angle:.2f} deg for {target_src}.")
            wcslist =  [build_wcs_centered_on(target_px, redf=redf, angle=angle) for target_px in [target_O, target_E]]
        else:
            logger.debug(f"Using {len(calibrators_in_field)} calibrators in field to fit WCS for {target_src}.")

            wcslist = list()

            _, (fits_O, fits_E) = zip(*[get_candidate_rank_by_matchs(redf, pos, angle=angle, r_search=30, calibrators=expected_sources_in_field) for pos in [target_O, target_E]])


            for img_label, target_px, fits in zip(["Ordinary", "Extraordinary"], [target_O, target_E], [fits_O, fits_E]):

                if len(fits) > 1:
                    # fit[0] is the astro source fitted, fit[1] (fit[1][0] is the gaussian, fit[1][1] is the constant

                    known_pos_src = [target_src]
                    known_pos_src.extend([fit[0] for fit in fits])

                    known_pos_px = [target_px]
                    known_pos_px.extend([(fit[1][0].x_mean.value, fit[1][0].y_mean.value) for fit in fits])

                    try:
                        with np.printoptions(precision=2, suppress=True):
                            for px, src in zip(known_pos_px, known_pos_src):
                                logger.debug(f"Fitting {img_label} {src.name} ({src.coord.ra.deg:.2f} ra, {src.coord.dec.deg:.2f} dec) to ({px[0]:.1f}, {px[1]:.1f})")

                        wcs_fitted = fit_wcs_from_points(np.array(known_pos_px).T, SkyCoord([src.coord for src in known_pos_src]), projection=build_wcs_centered_on(target_px, redf=redf, angle=angle))
                        wcslist.append(wcs_fitted)
                    except Exception as e:
                        logger.error(f"Exception {e} while fitting WCS, using pre-computed angle {angle:.2f} deg for {target_src}.")
                        wcslist.append(build_wcs_centered_on(target_px, redf=redf, angle=angle))
                else:
                    logger.debug("Using pre-computed wcs.")
                    wcslist.append(build_wcs_centered_on(target_px, redf=redf, angle=angle))

        if summary_kwargs['build_summary_images']:
            logger.debug(f"Building summary images for {redf}.")
            # plot summary of astrometry
            fig = mplt.figure.Figure(figsize=(6,6), dpi=iop4conf.mplt_default_dpi)
            ax = fig.subplots(nrows=1, ncols=1, subplot_kw={'projection': wcslist[0]})
            plot_preview_astrometry(redf, with_simbad=True, has_pairs=True, wcs1=wcslist[0], wcs2=wcslist[1], ax=ax, fig=fig) 
            fig.savefig(Path(redf.filedpropdir) / "astrometry_summary.png", bbox_inches="tight")
            fig.clear()


        gc.collect()

        return BuildWCSResult(success=True, wcslist=wcslist, info={})
    


    @classmethod
    def _build_wcs_for_polarimetry_from_target_O_and_E(cls, redf: 'ReducedFit', summary_kwargs : dict = None, n_seg_threshold=3.0, npixels=64, fwhm=None, centering=2/3) -> BuildWCSResult:
        r""" Deprecated. Build WCS for DIPOL polarimetry images by matching the found sources positions with the catalog.

        .. warning::
            This method is deprecated and will be removed in the future. It is kept here for reference. Do not use, unless you know what you are doing.

            When there are some DB sources in the field it probably works, but when there are not, it might give a result that is centered on the wrong source!
        
            It is safer to use the method _build_wcs_for_polarimetry_images_photo_quads, which uses the photometry field and quad matching to build the WCS.
            
        """

        if summary_kwargs is None:
            summary_kwargs = {'build_summary_images':True, 'with_simbad':True}

        # disp_allowed_err = 1.5*cls.disp_std
        disp_allowed_err =  np.array([30,30]) # most times should be much smaller (1.5*std)
        # but in bad cases, this is ~1 sigma of the gaussians

        from iop4lib.db import AstroSource
        from iop4lib.utils.sourcepairing import get_best_pairs
        from iop4lib.utils.quadmatching import distance

        logger.debug(f"{redf}: building WCS for DIPOL polarimetry images from target_O and target_E with {npixels=}, {n_seg_threshold=}.")
        
        # definitions

        target_src = redf.header_hintobject

        if target_src is None:
            raise Exception("No target source found in header, cannot build WCS.")
    
        data = redf.mdata
        
        # get the sources positions

        cx, cy = redf.width//2, redf.height//2
        positions = cls._estimate_positions_from_segments(redf=redf, n_seg_threshold=n_seg_threshold, npixels=npixels, fwhm=fwhm, centering=centering)

        if len(positions) == 0:
            logger.error(f"{redf}: Found no sources in the field, cannot build WCS.")
            return BuildWCSResult(success=False)

        if summary_kwargs['build_summary_images']:
            # plot summary of detected sources
            fig = mplt.figure.Figure(figsize=(6,6), dpi=iop4conf.mplt_default_dpi)
            ax = fig.subplots(nrows=1, ncols=1)
            imshow_w_sources(data, pos1=positions, ax=ax)
            candidates_aps = CircularAperture(positions[:2], r=10.0)
            candidates_aps.plot(ax, color="b")
            for i, (x,y) in enumerate(positions[:15]):
                ax.text(x, y, f"{i}", color="orange", fontdict={"size":14, "weight":"bold"})#, verticalalignment="center", horizontalalignment="center") 
            ax.plot([cx], [cy], '+', color='y', markersize=10)
            if centering:
                ax.axhline(cy-centering*redf.height/2, xmin=0, xmax=redf.width, color='y', linestyle='--')
                ax.axhline(cy+centering*redf.height/2, xmin=0, xmax=redf.width, color='y', linestyle='--')
                ax.axvline(cx-centering*redf.width/2, ymin=0, ymax=redf.height, color='y', linestyle='--')
                ax.axvline(cx+centering*redf.width/2, ymin=0, ymax=redf.height, color='y', linestyle='--')
            fig.savefig(Path(redf.filedpropdir) / "astrometry_detected_sources.png", bbox_inches="tight")
            fig.clear()
            
        if len(positions) == 1:
            logger.error(f"{redf}: Found only one source in the field, cannot build WCS.")
            return BuildWCSResult(success=False)
        
        if len(positions) > 2:
            logger.warning(f"{redf}: {len(positions)} sources found, expected 2. Maybe after looking at pairs only, we can find the right ones.")

            pre_list1, pre_list2 = zip(*itertools.product(positions, positions))

            # log some debug info about the pairs diference and the difference with respect the expected disp_sign_mean
            for i, (pos1, pos2) in enumerate(zip(pre_list1, pre_list2)):
                dist = distance(pos1, pos2)
                disp = np.abs(np.subtract(pos1, pos2))
                diff = np.abs(np.subtract(pos1, pos2))-np.abs(cls.disp_sign_mean)
                with np.printoptions(precision=1, suppress=True):
                    logger.debug(f"{i=}, {pos1=!s}, {pos2=!s}, dist={dist:.2f}, {disp=!s}, {diff=!s}")

            list1, list2, d0_new, disp_sign_new = get_best_pairs(pre_list1, pre_list2, cls.disp_sign_mean, disp_sign_err=disp_allowed_err)

            logger.debug(f"{list1=}, {list2=}, {d0_new=}, {disp_sign_new=}")

            if len(list1) == 0:
                list1, list2, d0_new, disp_sign_new = get_best_pairs(pre_list1, pre_list2, -cls.disp_sign_mean, disp_sign_err=disp_allowed_err)
                logger.debug(f"{list1=}, {list2=}, {d0_new=}, {disp_sign_new=}")
            
            if len(list1) != 1:
                logger.error(f"We expected exactly one source, but we found {len(list1)} pairs, returning success = False.")
                return BuildWCSResult(success=False)
            else:
                positions = [list1[0], list2[0]]
            

        # Check that the sources are pairs
        if not np.isclose(np.abs(positions[0][0]-positions[1][0]), cls.disp_mean[0], atol=disp_allowed_err[0]):
            logger.error(f"These are not pairs, x mismatch detected according to hard-coded pair distance: disp x = {np.abs(positions[0][0]-positions[1][0]):.0f} px")
            return BuildWCSResult(success=False, info={'n_bright_sources':len(positions)})
        if not np.isclose(np.abs(positions[0][1]-positions[1][1]), cls.disp_mean[1], atol=disp_allowed_err[1]):
            logger.error(f"These are not pairs, y mismatch detected according to hard-coded pair distance: disp y = {np.abs(positions[0][1]-positions[1][1]):.0f} px")
            return BuildWCSResult(success=False, info={'n_bright_sources':len(positions)})

        # define the targets to be the two positions found, with the ordinary on the right
        target_O, target_E = positions

        if target_O[0] < target_E[0]:
            target_O, target_E = target_E, target_O

        # get the right angle
        angle_mean, angle_std = get_angle_from_history(redf, target_src)
        logger.debug(f"Using angle {angle_mean:.2f} +- {angle_std:.2f} deg")

        # DIPOL polarimery images seem to be flipped vertically, which results in negative angle
        # TODO: watch this FLIP thing, check that indeed this is the behaviour
        if 'FLIPSTAT' in redf.rawfit.header:
            angle = - angle_mean
        else:
            angle = angle_mean

        # Build the WCS with the mean angle, and the right source in the center
        wcs1 = build_wcs_centered_on(target_O, target_src=target_src, redf=redf, angle=angle)
        wcs2 = build_wcs_centered_on(target_E, target_src=target_src, redf=redf, angle=angle)

        # TODO: try to improve wcs by finding other weaker sources and fitting a wcs with them
        # comparing them to a photometry field or DB or simbad

        if summary_kwargs['build_summary_images']:
            logger.debug(f"Building summary images for {redf}.")
            # plot summary of astrometry
            fig = mplt.figure.Figure(figsize=(6,6), dpi=iop4conf.mplt_default_dpi)
            ax = fig.subplots(nrows=1, ncols=1, subplot_kw={'projection': wcs1})
            plot_preview_astrometry(redf, with_simbad=True, has_pairs=True, wcs1=wcs1, wcs2=wcs2, ax=ax, fig=fig) 
            fig.savefig(Path(redf.filedpropdir) / "astrometry_summary.png", bbox_inches="tight")
            fig.clear()

        return BuildWCSResult(success=True, wcslist=[wcs1,wcs2],  info={'n_bright_sources':len(positions)})
    



    @classmethod
    def estimate_common_apertures(cls, reducedfits, reductionmethod=None, fit_boxsize=None, search_boxsize=(90,90)):
        aperpix, r_in, r_out, fit_res_dict = super().estimate_common_apertures(reducedfits, reductionmethod=reductionmethod, fit_boxsize=fit_boxsize, search_boxsize=search_boxsize, fwhm_min=5.0, fwhm_max=60)
        
        sigma = fit_res_dict['sigma']
        fwhm = fit_res_dict["mean_fwhm"]

        return 1.1*fwhm, 6*fwhm, 10*fwhm, fit_res_dict  

    @classmethod
    def get_instrumental_polarization(cls, reducedfit) -> dict:
        """ Returns the instrumental polarization for to be used for a given reducedfit.

        The instrumental polarization is a dictionary with the following keys:
            - Q_inst: instrumental Q Stokes parameter (0-1)
            - dQ_inst: instrumental Q error (0-1)
            - U_inst: instrumental U Stokes parameter (0-1)
            - dU_inst: instrumental U error (0-1)
            - CPA: zero-angle (deg)
        """

        if reducedfit.juliandate <= Time("2023-09-28 12:00").jd: # limpieza de espejos
            CPA = 44.5
            dCPA = 0.05
            Q_inst = 0.05777
            dQ_inst = 0.005
            U_inst = -3.77095
            dU_inst = 0.005
        else:
            CPA = 45.1
            dCPA = 0.05
            Q_inst = -0.0138 / 100
            dQ_inst = 0.005
            U_inst = -4.0806 / 100
            dU_inst = 0.005

        return {'Q_inst':Q_inst, 'dQ_inst':dQ_inst, 'U_inst':U_inst, 'dU_inst':dU_inst, 'CPA':CPA, 'dCPA':dCPA}


    @classmethod
    def compute_relative_polarimetry(cls, polarimetry_group):
        """ Computes the relative polarimetry for a polarimetry group for DIPOL
        """
        
        from iop4lib.db.aperphotresult import AperPhotResult
        from iop4lib.db.photopolresult import PhotoPolResult
        from iop4lib.utils import get_column_values

        # Perform some checks on the group

        if not all([reducedfit.instrument == INSTRUMENTS.DIPOL for reducedfit in polarimetry_group]):
            raise Exception(f"This method is only for DIPOL images.")

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
            logger.warning("Sources in field do not match for all polarimetry groups (ReducedFit %s): %s" % (",".join([str(redf.pk) for redf in polarimetry_group]), str(set.difference(*map(set, sources_in_field_qs_list)))))

        ## check rotation angles

        rot_angles_available = set([redf.rotangle for redf in polarimetry_group])
        rot_angles_required = {0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5}

        if not rot_angles_available.issubset(rot_angles_required):
            logger.warning(f"Rotation angles missing: {rot_angles_required - rot_angles_available}")

        if len(polarimetry_group) != 16:
            raise Exception(f"Can not compute relative polarimetry for a group with {len(polarimetry_group)} reducedfits, it should be 16.")

        # 1. Compute all aperture photometries

        aperpix, r_in, r_out, fit_res_dict = cls.estimate_common_apertures(polarimetry_group, reductionmethod=REDUCTIONMETHODS.RELPHOT)
        target_fwhm = fit_res_dict['mean_fwhm']
        
        logger.debug(f"Computing aperture photometries for the {len(polarimetry_group)} reducedfits in the group with target aperpix {aperpix:.1f}.")

        for reducedfit in polarimetry_group:
            cls.compute_aperture_photometry(reducedfit, aperpix, r_in, r_out)

        # 2. Compute relative polarimetry for each source (uses the computed aperture photometries)

        logger.debug("Computing relative polarimetry.")

        photopolresult_L = list()

        for astrosource in group_sources:

            if astrosource.calibrates.count() > 0:
                continue

            logger.debug(f"Computing relative polarimetry for {astrosource}.")

            # if any angle is missing for some pair, it uses the equivalent angle of the other pair

            aperphotresults = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, aperpix=aperpix, flux_counts__isnull=False)

            if len(aperphotresults) == 0:
                logger.error(f"No aperphotresults found for {astrosource}")
                continue

            if len(aperphotresults) != 32:
                logger.error(f"There should be 32 aperphotresults for each astrosource in the group, there are {len(aperphotresults)} for {astrosource.name}.")
                continue

            values = get_column_values(aperphotresults, ['reducedfit__rotangle', 'flux_counts', 'flux_counts_err', 'pairs'])

            angles_L = list(sorted(set(values['reducedfit__rotangle'])))

            if len(angles_L) != 16:
                logger.warning(f"There should be 16 different angles, there are {len(angles_L)}.")

            fluxD = {}
            for pair, angle, flux, flux_err in zip(values['pairs'], values['reducedfit__rotangle'], values['flux_counts'], values['flux_counts_err']):
                if pair not in fluxD:
                    fluxD[pair] = {}
                fluxD[pair][angle] = (flux, flux_err)

            # Dipol has the ordinary to the left and extraordinary to the right, IOP4 astrocalibration atm works the other way, swap them here

            F_O = np.array([(fluxD['E'][angle][0]) for angle in angles_L])
            dF_O = np.array([(fluxD['E'][angle][1]) for angle in angles_L])

            F_E = np.array([(fluxD['O'][angle][0]) for angle in angles_L])
            dF_E = np.array([(fluxD['O'][angle][1]) for angle in angles_L])

            N = len(angles_L)

            if astrosource.name == "2200+420":
                F_O = np.roll(F_E, 2)
                dF_O = np.roll(dF_E, 2)

            F = (F_O - F_E) / (F_O + F_E)
            dF = 2 / ( F_O + F_E )**2 * np.sqrt(F_E**2 * dF_O**2 + F_O**2 * dF_E**2)

            I = (F_O + F_E)
            dI = np.sqrt(dF_O**2 + dF_E**2)

            # Compute both the uncorrected and corrected values

            Qr_uncorr = 2/N * sum([F[i] * math.cos(math.pi/2*i) for i in range(N)])
            dQr_uncorr = 2/N * math.sqrt(sum([dF[i]**2 * math.cos(math.pi/2*i)**2 for i in range(N)]))

            logger.debug(f"{Qr_uncorr=}, {dQr_uncorr=}")

            Ur_uncorr = 2/N * sum([F[i] * math.sin(math.pi/2*i) for i in range(N)])
            dUr_uncorr = 2/N * math.sqrt(sum([dF[i]**2 * math.sin(math.pi/2*i)**2 for i in range(N)]))

            intrumental_polarization = cls.get_instrumental_polarization(reducedfit=polarimetry_group[0])
            Q_inst = intrumental_polarization['Q_inst']
            dQ_inst = intrumental_polarization['dQ_inst']
            U_inst = intrumental_polarization['U_inst']
            dU_inst = intrumental_polarization['dU_inst']
            CPA = intrumental_polarization['CPA']
            dCPA = intrumental_polarization['dCPA']

            logger.debug(f"{Q_inst=}, {dQ_inst=}")
            logger.debug(f"{U_inst=}, {dU_inst=}")

            Qr = Qr_uncorr - Q_inst
            dQr = math.sqrt(dQr_uncorr**2 + dQ_inst**2)

            logger.debug(f"{Qr=}, {dQr=}")

            Ur = Ur_uncorr - U_inst
            dUr = math.sqrt(dUr_uncorr**2 + dU_inst**2)

            logger.debug(f"{Ur=}, {dUr=}")

            def _get_p_and_chi(Qr, Ur, dQr, dUr):
                # linear polarization (0 to 1)
                P = math.sqrt(Qr**2+Ur**2)
                dP = 1/P * math.sqrt((Qr*dQr)**2 + (Ur*dUr)**2)

                # polarization angle (degrees)
                chi = 0.5 * math.degrees(math.atan2(Ur, Qr))
                dchi = 0.5 * math.degrees( 1 / (Qr**2 + Ur**2) * math.sqrt((Qr*dUr)**2 + (Ur*dQr)**2) )

                return P, chi, dP, dchi
            
            P_uncorr, chi_uncorr, dP_uncorr, dchi_uncorr = _get_p_and_chi(Qr_uncorr, Ur_uncorr, dQr_uncorr, dUr_uncorr)
            P, chi, dP, dchi = _get_p_and_chi(Qr, Ur, dQr, dUr)
            chi = chi + CPA
            dchi = math.sqrt(dchi**2 + dCPA**2)

            logger.debug(f"{P=}, {chi=}, {dP=}, {dchi=}")

            # No attempt to get magnitude from polarimetry fields in dipol, they have too low exposure, and many times there are no calibrators in the subframe.

            # save the results
                    
            result = PhotoPolResult.create(reducedfits=polarimetry_group, 
                                                            astrosource=astrosource, 
                                                            reduction=REDUCTIONMETHODS.RELPOL, 
                                                            p=P, p_err=dP, chi=chi, chi_err=dchi,
                                                            _q_nocorr=Qr_uncorr, _u_nocorr=Ur_uncorr, _p_nocorr=P_uncorr, _chi_nocorr=chi_uncorr,
                                                            aperpix=aperpix)
            
            photopolresult_L.append(result)

        # 3. Save results
        for result in photopolresult_L:
            result.save()
