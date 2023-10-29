# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
import os
import re
import astrometry
import numpy as np
import astropy.units as u

# iop4lib imports
from iop4lib.enums import *
from .instrument import Instrument

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

    required_masters = ['masterbias', 'masterflat', 'masterdark']

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
        elif rawfit.header['FILTER'] == "Red":  
            rawfit.band = BANDS.R
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
                if not source['other_name']:
                    continue
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
    def build_wcs(cls, reducedfit: 'ReducedFit', build_summary_images : bool = True, summary_kwargs : dict = {'with_simbad':True}):
        """ Overriden Instrument build_wcs.
        
        While for PHOTOMETRY observations, DIPOL has a wide field which can be astrometrically calibrated, 
        POLARIMETRY files are small with only the source field ordinary and extraordianty images in the center (to save up space).
        In some ocassions, there might be some close source also in the field.

        Therefore, to calibrate polarimetry files, we just give it a WCS centered on the source.

        For PHOTOMETRY files, we use the parent class method, but we set some custom shotgun_params_kwargs to account
        for the low flux and big size of the images.

        """
    
        if reducedfit.obsmode == OBSMODES.PHOTOMETRY:
            return super().build_wcs(reducedfit, shotgun_params_kwargs=cls._build_shotgun_params(reducedfit), build_summary_images=build_summary_images, summary_kwargs=summary_kwargs)
        elif reducedfit.obsmode == OBSMODES.POLARIMETRY:
            return cls._build_wcs_for_polarimetry_images_photo_quads(reducedfit, build_summary_images=build_summary_images, summary_kwargs=summary_kwargs)
        else:
            logger.error(f"Unknown obsmode {reducedfit.obsmode} for {reducedfit}.")
            
        
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
        shotgun_params_kwargs["bins"] = [400]
        shotgun_params_kwargs["hist_range"] = [(0,500)]

        shotgun_params_kwargs["position_hint"] = [redf.get_astrometry_position_hint(allsky=False)]
        shotgun_params_kwargs["size_hint"] = [redf.get_astrometry_size_hint()]

        return shotgun_params_kwargs


    @classmethod
    def _build_wcs_for_polarimetry_images_photo_quads(cls, redf: 'ReducedFit', build_summary_images : bool = True, summary_kwargs : dict = {'with_simbad':True}):
        
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
            raise Exception(f"No astro-calibrated photometry field found for {redf_pol}.")

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

            fwhm = 4
            kernel_size = 2*int(fwhm)+1
            kernel = make_2dgaussian_kernel(fwhm, size=kernel_size)
            data = convolve(data, kernel)

            mean, median, std = sigma_clipped_stats(data, sigma=5.0)

            daofind = DAOStarFinder(fwhm=30.0, threshold=3*std, brightest=100)  
            sources = daofind(data - median)
            sources.sort('flux', reverse=True)

            sources = sources[:10]
            
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

            sets_L.append(positions)

        if build_summary_images:
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

        from iop4lib.utils.quadmatching import hash_juan, distance, order, qorder_juan
        hash_func = hash_juan
        hashes_1 = np.array([hash_func(quad) for quad in quads_1])
        hashes_2 = np.array([hash_func(quad) for quad in quads_2])

        all_indices = np.array(list(itertools.product(range(len(quads_1)),range(len(quads_2)))))
        all_distances = np.array([distance(hashes_1[i], hashes_2[j]) for i,j in all_indices])

        idx = np.argsort(all_distances)
        all_indices = all_indices[idx]
        all_distances = all_distances[idx]

        if build_summary_images:
            colors = [color for color in mplt.rcParams["axes.prop_cycle"].by_key()["color"]]

            fig = mplt.figure.Figure(figsize=(12,6), dpi=iop4conf.mplt_default_dpi)
            axs = fig.subplots(nrows=1, ncols=2)

            for (i, j), color in list(zip(all_indices, colors))[:1]: 

                for ax, data, quad, positions in zip(axs, [redf_pol.mdata, photdata_subframe], [quads_1[i], quads_2[j]], sets_L):
                    imshow_w_sources(data, pos1=positions, ax=ax)
                    x, y = np.array(order(quad)).T
                    ax.fill(x, y, edgecolor='k', fill=True, facecolor=mplt.colors.to_rgba(color, alpha=0.2))
                    for pi, p in enumerate(np.array((qorder_juan(quad)))):
                        xp = p[0]
                        yp = p[1]
                        ax.text(xp, yp, f"{pi}", fontsize=16, color="y")

                fig.suptitle(f"dist({i},{j})={distance(hash_func(quads_1[i]),hash_func(quads_2[j])):.3f}", y=0.83)

            axs[0].set_title("Polarimetry")
            axs[1].set_title("Photometry")
            
            fig.savefig(Path(redf.filedpropdir) / "astrometry_matched_quads.png", bbox_inches="tight")
            fig.clf()

        # Build the WCS
        quads_1 = [qorder_juan(quad) for quad in quads_1]
        quads_2 = [qorder_juan(quad) for quad in quads_2]

        angle_mean, angle_std = get_angle_from_history(redf, target_src)
        if 'FLIPSTAT' in redf.rawfit.header and 'FLIP' in redf.rawfit.header['FLIPSTAT'].upper():
            angle = - angle_mean
        else:
            angle = angle_mean

        best_i, best_j = all_indices[0]

        pre_wcs = build_wcs_centered_on((redf_pol.width//2,redf_pol.height//2), redf=redf_phot, angle=angle)
        
        pix_array_1 = np.array(list(zip(*[(x,y) for x,y in quads_1[best_i]])))
        pix_array_2 = np.array(list(zip(*[(x+x_start,y+y_start) for x,y in quads_2[best_j]])))

        wcs1 = fit_wcs_from_points(pix_array_1,  redf_phot.wcs1.pixel_to_world(*pix_array_2), projection=pre_wcs)
        wcs2 = fit_wcs_from_points(pix_array_1,  redf_phot.wcs2.pixel_to_world(*pix_array_2), projection=pre_wcs)

        wcslist = [wcs1, wcs2]

        if build_summary_images:
            fig = mplt.figure.Figure(figsize=(6,6), dpi=iop4conf.mplt_default_dpi)
            ax = fig.subplots(nrows=1, ncols=1, subplot_kw={'projection': wcslist[0]})
            plot_preview_astrometry(redf_pol, with_simbad=True, has_pairs=True, wcs1=wcslist[0], wcs2=wcslist[1], ax=ax, fig=fig) 
            fig.savefig(Path(redf_pol.filedpropdir) / "astrometry_summary.png", bbox_inches="tight")
            fig.clear()            

        return BuildWCSResult(success=True, wcslist=wcslist, info={'redf_phot__pk':redf_phot.pk, 'redf_phot__fileloc':redf_phot.fileloc})



    @classmethod
    def get_header_hintcoord(cls, rawfit):
        """ Overriden for DIPOL

        As of 2023-10-23, DIPOL does not inclide RA and DEC in the header, RA and DEC will be derived from the object name.
        """     
        
        return rawfit.header_objecthint.coord