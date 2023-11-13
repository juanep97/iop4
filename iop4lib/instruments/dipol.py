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
import itertools
import datetime
import math

# iop4lib imports
from iop4lib.enums import *
from .instrument import Instrument
from iop4lib.utils import imshow_w_sources, get_candidate_rank_by_matchs, get_angle_from_history, build_wcs_centered_on, get_simbad_sources
from iop4lib.utils.sourcedetection import get_sources_daofind, get_segmentation, get_cat_sources_from_segment_map, get_bkg
from iop4lib.utils.plotting import plot_preview_astrometry
from iop4lib.utils.astrometry import BuildWCSResult


# logging
import logging
logger = logging.getLogger(__name__)


import typing
if typing.TYPE_CHECKING:
    from iop4lib.db import RawFit, ReducedFit, Epoch

class DIPOL(Instrument):

    name = "DIPOL"
    instrument_kw = "ASI Camera (1)"
    
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

        master = model.objects.filter(**args).first()
        
        if master is None and other_epochs == True:
            args.pop("epoch")

            master_other_epochs = np.array(model.objects.filter(**args).all())

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
        mf_data = fits.getdata(reducedfit.masterflat.filepath)[idx]

        if reducedfit.masterdark is not None:
            md_dark = fits.getdata(reducedfit.masterdark.filepath)[idx]
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
    def get_astrometry_position_hint(cls, rawfit: 'RawFit', allsky=False, n_field_width=1.5):
        """ Get the position hint from the FITS header as an astrometry.PositionHint."""        

        hintcoord = cls.get_header_hintcoord(rawfit)

        if rawfit.header["XBINNING"] != 2:
            logger.error(f"Cannot compute astrometry for {rawfit} because of the binning: {rawfit.header['XBINNING']}.")
            return None
        
        if allsky:
            hintsep = 180.0
        else:
            hintsep = (n_field_width * cls.field_width_arcmin*u.Unit("arcmin")).to_value(u.deg)

        return astrometry.PositionHint(ra_deg=hintcoord.ra.deg, dec_deg=hintcoord.dec.deg, radius_deg=hintsep)
    
    @classmethod
    def has_pairs(cls, fit_instance: 'ReducedFit' or 'RawFit') -> bool:
        """ DIPOL ALWAYS HAS PAIRS?!!!! """
        return True





    @classmethod
    def _estimate_positions_from_segments(cls, redf, fwhm=None, npixels=64, n_seg_threshold=3.0, centered=True):

        if redf.header_hintobject.srctype == SRCTYPES.STAR and redf.exptime <= 5:
            fwhm = 80.0
        else:
            fwhm = 1.0

        # get the sources positions

        data = redf.data

        mean, median, std = sigma_clipped_stats(data, sigma=5.0)

        bkg = get_bkg(redf.mdata, filter_size=5, box_size=redf.width//10)
        imgdata_bkg_substracted = redf.mdata - bkg.background
        seg_threshold = n_seg_threshold * bkg.background_rms
        
        segment_map, convolved_data = get_segmentation(imgdata_bkg_substracted, fwhm=fwhm, npixels=npixels, threshold=seg_threshold)
        if segment_map is None:
            return list()
        else:
            seg_cat, positions, tb = get_cat_sources_from_segment_map(segment_map, imgdata_bkg_substracted, convolved_data)
        
        if centered:
            # select only the sources in the center
            cx, cy = redf.width//2, redf.height//2
            idx = np.abs(positions[:,0]-cx) < 1/3 * redf.width
            idx = idx & (np.abs(positions[:,1]-cy) < 1/3 * redf.height)
            positions = positions[idx]

        return positions

    @classmethod
    def build_wcs(cls, reducedfit: 'ReducedFit', summary_kwargs : dict = {'build_summary_images':True, 'with_simbad':True}, method=None):
        """ Overriden Instrument build_wcs.
        
        While for PHOTOMETRY observations, DIPOL has a wide field which can be astrometrically calibrated, 
        POLARIMETRY files are small with only the source field ordinary and extraordianty images in the center (to save up space).
        In some ocassions, there might be some close source also in the field.

        Therefore, to calibrate polarimetry files, we just give it a WCS centered on the source.

        For PHOTOMETRY files, we use the parent class method, but we set some custom shotgun_params_kwargs to account
        for the low flux and big size of the images.

        """

        if method is not None:
            logger.warning(f"Calling {method} for {reducedfit}.")
            return method(reducedfit, summary_kwargs=summary_kwargs)

        from iop4lib.db import ReducedFit, AstroSource

        target_src = reducedfit.header_hintobject

        if reducedfit.obsmode == OBSMODES.PHOTOMETRY:
            return super().build_wcs(reducedfit, shotgun_params_kwargs=cls._build_shotgun_params(reducedfit), summary_kwargs=summary_kwargs)
        elif reducedfit.obsmode == OBSMODES.POLARIMETRY:

            # Gather some info to perform a good decision on which methods to use

            n_estimate = len(cls._estimate_positions_from_segments(redf=reducedfit, n_seg_threshold=1.5, centered=False))
            n_estimate_centered = len(cls._estimate_positions_from_segments(redf=reducedfit, n_seg_threshold=1.5, centered=True))
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

            def _try_EO_method():

                if target_src.srctype == SRCTYPES.STAR:
                    n_seg_threshold_L = [300, 200, 100, 50, 25, 12, 6]
                    if reducedfit.exptime <= 5:
                        npixels_L = [128, 256, 64]
                    else:
                        npixels_L = [64, 128]
                else:
                    n_seg_threshold_L = [6.0, 3.0, 1.5, 1.0]
                    npixels_L = [64]

                for npixels, n_seg_threshold in itertools.product(npixels_L, n_seg_threshold_L):
                    if (build_wcs_result := cls._build_wcs_for_polarimetry_from_target_O_and_E(reducedfit, summary_kwargs=summary_kwargs, n_seg_threshold=n_seg_threshold, npixels=npixels)):
                        break
                return build_wcs_result
                            
            def _try_quad_method():
                if redf_phot is not None:
                    
                    if target_src.srctype == SRCTYPES.STAR:
                        n_threshold_L = [300, 200, 100, 50, 25, 12, 6]
                    else:
                        n_threshold_L = [15,5,3]

                    for fwhm, n_threshold in itertools.product([30,15], n_threshold_L):
                        if (build_wcs_result := cls._build_wcs_for_polarimetry_images_photo_quads(reducedfit, summary_kwargs=summary_kwargs, n_threshold=n_threshold, find_fwhm=fwhm, smooth_fwhm=4)):
                            break
                else:
                    build_wcs_result = BuildWCSResult(success=False)
                return build_wcs_result
            
            def _try_catalog_method():

                if target_src.srctype == SRCTYPES.STAR:
                    n_seg_threshold_L = [700, 500, 400, 300, 200, 100, 50]
                    npixels_L = [128, 64]
                else:
                    n_seg_threshold_L = [1.0]
                    npixels_L = [64, 32]
            
                if n_expected_calibrators > 0 or n_expected_simbad_sources > 0:
                    for npixels, n_seg_threshold in itertools.product(npixels_L, n_seg_threshold_L):
                        if (build_wcs := cls._build_wcs_for_polarimetry_images_catalog_matching(reducedfit, summary_kwargs=summary_kwargs, n_seg_threshold=n_seg_threshold, npixels=npixels)):
                            break
                else:
                    build_wcs = BuildWCSResult(success=False)
                return build_wcs   
            

            method_try_order = [_try_EO_method, _try_quad_method, _try_catalog_method]

            if target_src.srctype == SRCTYPES.STAR:
                method_try_order = [_try_EO_method, _try_quad_method, _try_catalog_method]
            else:
                method_try_order = [_try_quad_method, _try_catalog_method, _try_EO_method]

            for m in method_try_order:
                logger.debug(f"Trying {m.__name__} for {reducedfit}.")
                if (build_wcs := m()):
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
    def _build_wcs_for_polarimetry_images_photo_quads(cls, redf: 'ReducedFit', summary_kwargs : dict = {'build_summary_images':True, 'with_simbad':True}, n_threshold=5.0, find_fwhm=30, smooth_fwhm=4):
        
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

        # get the subframe of the photometry field that corresponds to this polarimetry field, (approx)
        x_start = redf_pol.rawfit.header['XORGSUBF']
        y_start = redf_pol.rawfit.header['YORGSUBF']

        x_end = x_start + redf_pol.rawfit.header['NAXIS1']
        y_end = y_start + redf_pol.rawfit.header['NAXIS2']

        idx = np.s_[y_start:y_end, x_start:x_end]

        photdata_subframe = redf_phot.mdata[idx] # if we use the hash_juan_old, which is not invariant under fliiping, we need to flip the image in y (redf_phot.mdata[idx][::-1,:])

        # find 10 brightest sources in each field

        sets_L = list()

        for data in [redf_pol.mdata, photdata_subframe]:

            if smooth_fwhm:
                kernel_size = 2*int(smooth_fwhm)+1
                kernel = make_2dgaussian_kernel(smooth_fwhm, size=kernel_size)
                data = convolve(data, kernel)

            mean, median, std = sigma_clipped_stats(data, sigma=5.0)

            daofind = DAOStarFinder(fwhm=find_fwhm, threshold=n_threshold*std, brightest=100, exclude_border=True)  
            sources = daofind(data - median)

            if len(sources) < 4:
                return BuildWCSResult(success=False)

            sources.sort('flux', reverse=True)

            sources = sources[:10]
            
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

            sets_L.append(positions)

        if summary_kwargs['build_summary_images']:
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
            fig.savefig(Path(redf.filedpropdir) / "astrometry_detected_sources.png", bbox_inches="tight")
            fig.clf()

        # Build the quads for each field
        quads_1 = np.array(list(itertools.combinations(sets_L[0], 4)))
        quads_2 = np.array(list(itertools.combinations(sets_L[1], 4)))

        from iop4lib.utils.quadmatching import hash_juan, distance, order, qorder_juan, find_linear_transformation
        hash_func, qorder = hash_juan, qorder_juan

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
 
        idx_selected = np.where(all_distances < 4.0)[0] 
        indices_selected = all_indices[idx_selected]
        distances_selected = all_distances[idx_selected]
        
        if np.sum(idx_selected) == 0:
            logger.error(f"No quads with distance < 4.0, returning success = False.")
            return BuildWCSResult(success=False, wcslist=None, info={'redf_phot__pk':redf_phot.pk, 'redf_phot__fileloc':redf_phot.fileloc}) 
        else:
            idx_selected = np.argsort(distances_selected)[:5]
            indices_selected = all_indices[idx_selected]
            distances_selected = all_distances[idx_selected]

        # get linear transforms
        logger.debug(f"Selected {len(indices_selected)} quads with distance < 4.0. I will get the one with less deviation from the median linear transform.")

        R_L, t_L = zip(*[find_linear_transformation(qorder(quads_1[i]), qorder(quads_2[j])) for i,j in indices_selected])
        logger.debug(f"{t_L=}")
        
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

                for ax, data, quad, positions in zip(axs, [redf_pol.mdata, photdata_subframe], [quads_1[i], quads_2[j]], sets_L):
                    imshow_w_sources(data, pos1=positions, ax=ax)
                    x, y = np.array(order(quad)).T
                    ax.fill(x, y, edgecolor='k', fill=True, facecolor=mplt.colors.to_rgba(color, alpha=0.2))
                    for pi, p in enumerate(np.array((qorder(quad)))):
                        xp = p[0]
                        yp = p[1]
                        ax.text(xp, yp, f"{pi}", fontsize=16, color="y")

                fig.suptitle(f"dist({i},{j})={distance(hash_func(quads_1[i]),hash_func(quads_2[j])):.3f}", y=0.83)

            axs[0].set_title("Polarimetry")
            axs[1].set_title("Photometry")
            
            fig.savefig(Path(redf.filedpropdir) / "astrometry_matched_quads.png", bbox_inches="tight")
            fig.clf()

        # Build the WCS

        # give an unique ordering to the quads

        quads_1 = [qorder_juan(quad) for quad in quads_1]
        quads_2 = [qorder_juan(quad) for quad in quads_2]

        # get the pre wcs with the target in the center of the image

        angle_mean, angle_std = get_angle_from_history(redf, target_src)
        if 'FLIPSTAT' in redf.rawfit.header: 
            # TODO: check that indeed the mere presence of this keyword means that the image is flipped, without the need of checking the value. 
            # FLIPSTAT is a MaximDL thing only, but it seems that the iamge is flipped whenever the keyword is present, regardless of the value.
            angle = - angle_mean
        else:
            angle = angle_mean

        pre_wcs = build_wcs_centered_on((redf_pol.width//2,redf_pol.height//2), redf=redf_phot, angle=angle)
        
        # get the pixel arrays in the polarimetry field and in the FULL photometry field to relate them
        pix_array_1 = np.array(list(zip(*[(x,y) for x,y in quads_1[best_i]])))
        pix_array_2 = np.array(list(zip(*[(x+x_start,y+y_start) for x,y in quads_2[best_j]])))

        # fit the WCS so the pixel arrays in 1 correspond to the ra/dec of the pixel array in 2
        wcs1 = fit_wcs_from_points(pix_array_1,  redf_phot.wcs1.pixel_to_world(*pix_array_2), projection=pre_wcs)
        wcs2 = fit_wcs_from_points(pix_array_1,  redf_phot.wcs2.pixel_to_world(*pix_array_2), projection=pre_wcs)

        wcslist = [wcs1, wcs2]

        if summary_kwargs['build_summary_images']:
            fig = mplt.figure.Figure(figsize=(6,6), dpi=iop4conf.mplt_default_dpi)
            ax = fig.subplots(nrows=1, ncols=1, subplot_kw={'projection': wcslist[0]})
            plot_preview_astrometry(redf_pol, with_simbad=True, has_pairs=True, wcs1=wcslist[0], wcs2=wcslist[1], ax=ax, fig=fig) 
            fig.savefig(Path(redf_pol.filedpropdir) / "astrometry_summary.png", bbox_inches="tight")
            fig.clear()            


        result = BuildWCSResult(success=True, wcslist=wcslist, info={'method':'_build_wcs_for_polarimetry_images_photo_quads', 'redf_phot__pk':redf_phot.pk, 'redf_phot__fileloc':redf_phot.fileloc, 'smooth_fwhm':smooth_fwhm, 'n_threshold':n_threshold, 'find_fwhm':find_fwhm})

        return result



    @classmethod
    def _build_wcs_for_polarimetry_images_catalog_matching(cls, redf: 'ReducedFit', summary_kwargs : dict = {'build_summary_images':True, 'with_simbad':True}, n_seg_threshold=1.5, npixels=64):
        r""" Deprecated. Build WCS for DIPOL polarimetry images by matching the found sources positions with the catalog.

        .. warning::
            This method is deprecated and will be removed in the future. It is kept here for reference. Do not use, unless you know what you are doing.

            When there are some DB sources in the field it probably works, but when there are not, it might give a result that is centered on the wrong source!
        
            It is safer to use the method _build_wcs_for_polarimetry_images_photo_quads, which uses the photometry field and quad matching to build the WCS.
            
        """

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

        positions = cls._estimate_positions_from_segments(redf, n_seg_threshold=n_seg_threshold, npixels=npixels, centered=True)
        positions_non_centered = cls._estimate_positions_from_segments(redf, n_seg_threshold=n_seg_threshold, npixels=npixels, centered=False)

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
            ax.axhline(cy-1/3*redf.height, xmin=0, xmax=redf.width, color='y', linestyle='--')
            ax.axhline(cy+1/3*redf.height, xmin=0, xmax=redf.width, color='y', linestyle='--')
            ax.axvline(cx-1/3*redf.width, ymin=0, ymax=redf.height, color='y', linestyle='--')
            ax.axvline(cx+1/3*redf.width, ymin=0, ymax=redf.height, color='y', linestyle='--')
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
                        logger.debug(f"{i=},\t{pos1=!s},\t{pos2=!s},\t{dist=!s},\t{disp=!s},\t{diff=!s}")

                # get the best pairs according to the disp_sign_mean 
                # since we dont know if pre_list1 is the ordinary or extraordinary image, try with 
                # disp_sign_mean and -disp_sign_mean

                list1, list2, d0_new, disp_sign_new = get_best_pairs(pre_list1, pre_list2, cls.disp_sign_mean, disp_sign_err=disp_allowed_err)
                logger.debug(f"{list1=}, {list2=}, {d0_new=}, {disp_sign_new=}")
                if len(list1) == 0:
                    list1, list2, d0_new, disp_sign_new = get_best_pairs(pre_list1, pre_list2, -cls.disp_sign_mean, disp_sign_err=disp_allowed_err)
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

        logger.debug(f"Found {len(calibrators_in_field)} calibrators in field for {target_src}")

        if len(calibrators_in_field) <= 1 or len(positions) <= 2:
            logger.warning(f"Using pre-computed angle {angle:.2f} deg for {target_src}.")
            wcslist = [build_wcs_centered_on(target_px, redf=redf, angle=angle) for target_px in [target_O, target_E]]
        else:
            logger.debug(f"Using {len(calibrators_in_field)} calibrators in field to fit WCS for {target_src}.")

            wcslist = list()

            _, (fits_O, fits_E) = zip(*[get_candidate_rank_by_matchs(redf, pos, angle=angle, r_search=30, calibrators=expected_sources_in_field) for pos in [target_O, target_E]])

            for target_px, fits in zip([target_O, target_E], [fits_O, fits_E]):
                known_pos_skycoord = [target_src.coord]
                # fit[0] is the astro source fitted, fit[1] (fit[1][0] is the gaussian, fit[1][1] is the constant
                known_pos_skycoord.extend([fit[0].coord for fit in fits])

                known_pos_px = [target_px]
                known_pos_px.extend([(fit[1][0].x_mean.value, fit[1][0].y_mean.value) for fit in fits])

                try:
                    logger.debug("Fitting " + ", ".join([f"ra {coord.ra.deg} dec {coord.dec.deg} to {pos}" for coord, pos in zip(known_pos_skycoord, known_pos_px)]))

                    wcs_fitted = fit_wcs_from_points(np.array(known_pos_px).T, SkyCoord(known_pos_skycoord), projection=build_wcs_centered_on(target_px, redf=redf, angle=angle))
                    wcslist.append(wcs_fitted)
                except Exception as e:
                    logger.error(f"Exception {e} while fitting WCS, using pre-computed angle {angle:.2f} deg for {target_src}.")
                    wcslist = [build_wcs_centered_on(target_px, redf=redf, angle=angle) for target_px in [target_O, target_E]]
                


        if summary_kwargs['build_summary_images']:
            logger.debug(f"Building summary images for {redf}.")
            # plot summary of astrometry
            fig = mplt.figure.Figure(figsize=(6,6), dpi=iop4conf.mplt_default_dpi)
            ax = fig.subplots(nrows=1, ncols=1, subplot_kw={'projection': wcslist[0]})
            plot_preview_astrometry(redf, with_simbad=True, has_pairs=True, wcs1=wcslist[0], wcs2=wcslist[1], ax=ax, fig=fig) 
            fig.savefig(Path(redf.filedpropdir) / "astrometry_summary.png", bbox_inches="tight")
            fig.clear()

        return BuildWCSResult(success=True, wcslist=wcslist, info={'method':'_build_wcs_for_polarimetry_images_catalog_matching'})
    


    @classmethod
    def _build_wcs_for_polarimetry_from_target_O_and_E(cls, redf: 'ReducedFit', summary_kwargs : dict = {'build_summary_images':True, 'with_simbad':True}, n_seg_threshold=3.0, npixels=64) -> BuildWCSResult:
        r""" Deprecated. Build WCS for DIPOL polarimetry images by matching the found sources positions with the catalog.

        .. warning::
            This method is deprecated and will be removed in the future. It is kept here for reference. Do not use, unless you know what you are doing.

            When there are some DB sources in the field it probably works, but when there are not, it might give a result that is centered on the wrong source!
        
            It is safer to use the method _build_wcs_for_polarimetry_images_photo_quads, which uses the photometry field and quad matching to build the WCS.
            
        """

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
        positions = cls._estimate_positions_from_segments(redf, n_seg_threshold=n_seg_threshold, npixels=npixels, centered=True)

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
            ax.axhline(cy-1/3*redf.height, xmin=0, xmax=redf.width, color='y', linestyle='--')
            ax.axhline(cy+1/3*redf.height, xmin=0, xmax=redf.width, color='y', linestyle='--')
            ax.axvline(cx-1/3*redf.width, ymin=0, ymax=redf.height, color='y', linestyle='--')
            ax.axvline(cx+1/3*redf.width, ymin=0, ymax=redf.height, color='y', linestyle='--')
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
                    logger.debug(f"{i=}, {pos1=!s}, {pos2=!s}, {dist=!s}, {disp=!s}, {diff=!s}")

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
        return 1.8*sigma, 5*sigma, 10*sigma, fit_res_dict
  

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
            logger.warning(f"Sources in field do not match for all polarimetry groups: {set.difference(*map(set, sources_in_field_qs_list))}")

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
            logger.debug(f"Computing relative polarimetry for {astrosource}.")

            # if any angle is missing for some pair, it uses the equivalent angle of the other pair

            aperphotresults = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, aperpix=aperpix, flux_counts__isnull=False)

            if len(aperphotresults) == 0:
                logger.error(f"No aperphotresults found for {astrosource}")
                continue

            if len(aperphotresults) != 32:
                logger.warning(f"There should be 32 aperphotresults for each astrosource in the group, there are {len(aperphotresults)}.")

            values = get_column_values(aperphotresults, ['reducedfit__rotangle', 'flux_counts', 'flux_counts_err', 'pairs'])

            angles_L = list(sorted(set(values['reducedfit__rotangle'])))
            if len(angles_L) != 16:
                logger.warning(f"There should be 16 different angles, there are {len(angles_L)}.")

            fluxD = {}
            for pair, angle, flux, flux_err in zip(values['pairs'], values['reducedfit__rotangle'], values['flux_counts'], values['flux_counts_err']):
                if pair not in fluxD:
                    fluxD[pair] = {}
                fluxD[pair][angle] = (flux, flux_err)

            F_O = np.array([(fluxD['O'][angle][0]) for angle in angles_L])
            dF_O = np.array([(fluxD['O'][angle][1]) for angle in angles_L])

            F_E = np.array([(fluxD['E'][angle][0]) for angle in angles_L])
            dF_E = np.array([(fluxD['E'][angle][1]) for angle in angles_L])

            F = (F_O - F_E) / (F_O + F_E)
            dF = 1 / ( F_O**2 + F_E**2 ) * np.sqrt(dF_O**2 + dF_E**2)

            I = (F_O + F_E)
            dI = np.sqrt(dF_O**2 + dF_E**2)

            N = len(angles_L)

            # Compute both the uncorrected and corrected values

            Qr_uncorr = 2/N * sum([F[i] * math.cos(math.pi/2*i) for i in range(N)])
            dQr_uncorr = 2/N * math.sqrt(sum([dF[i]**2 * math.cos(math.pi/2*i)**2 for i in range(N)]))

            logger.debug(f"{Qr_uncorr=}, {dQr_uncorr=}")

            Ur_uncorr = 2/N * sum([F[i] * math.sin(math.pi/2*i) for i in range(N)])
            dUr_uncorr = 2/N * math.sqrt(sum([dF[i]**2 * math.sin(math.pi/2*i)**2 for i in range(N)]))


            Q_inst = +0.057/100
            dQ_inst = 0

            logger.debug(f"{Q_inst=}, {dQ_inst=}")

            U_inst = -3.77/100
            dU_inst = 0

            Qr = Qr_uncorr + Q_inst # TODO: check and derive this value 
            dQr = math.sqrt(dQr_uncorr**2 + dQ_inst**2)

            logger.debug(f"{Qr=}, {dQr=}")

            Ur = Ur_uncorr + U_inst # TODO: check and derive this value
            dUr = math.sqrt(dUr_uncorr**2 + dU_inst**2)

            logger.debug(f"{Ur=}, {dUr=}")

            def _get_p_and_chi(Qr, Ur, dQr, dUr):
                # linear polarization (0 to 1)
                P = math.sqrt(Qr**2+Ur**2)
                dP = 1/P * math.sqrt((Qr*dQr)**2 + (Ur*dUr)**2)
                # polarization angle (degrees)
                x = -Qr/Ur
                dx = math.sqrt( (-1/Ur)**2+dUr**2 + (+Qr/Ur**2)**2*dQr**2 )
                chi = 0.5 * math.degrees(math.atan2(-Qr, Ur))
                dchi = 0.5 * 1/(1 + x**2) * dx

                return P, chi, dP, dchi
            
            # linear polarization (0 to 1)
            P_uncorr, chi_uncorr, dP_uncorr, dchi_uncorr = _get_p_and_chi(Qr_uncorr, Ur_uncorr, dQr_uncorr, dUr_uncorr)
            P, chi, dP, dchi = _get_p_and_chi(Qr, Ur, dQr, dUr)

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
