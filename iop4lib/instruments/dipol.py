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
    def get_header_objecthint(self, rawfit):
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
    def build_wcs(self, reducedfit: 'ReducedFit'):
        """ Overriden Instrument build_wcs.
        
        While for PHOTOMETRY observations, DIPOL has a wide field which can be astrometrically calibrated, 
        POLARIMETRY files are small with only the source field ordinary and extraordianty images in the center (to save up space).
        In some ocassions, there might be some close source also in the field.

        Therefore, to calibrate polarimetry files, we just give it a WCS centered on the source.

        For PHOTOMETRY files, we use the parent class method, but we set some custom shotgun_params_kwargs to account
        for the low flux and big size of the images.

        """
    
        if reducedfit.obsmode == OBSMODES.PHOTOMETRY:

            shotgun_params_kwargs = dict()

            shotgun_params_kwargs["keep_n_seg"] = [300]
            shotgun_params_kwargs["border_margin_px"] = [20]
            shotgun_params_kwargs["output_logodds_threshold"] = [14]
            shotgun_params_kwargs["n_rms_seg"] = [1.5, 1.2, 1.0]
            shotgun_params_kwargs["bkg_filter_size"] = [11] 
            shotgun_params_kwargs["bkg_box_size"] = [32]
            shotgun_params_kwargs["seg_fwhm"] = [1.0]
            shotgun_params_kwargs["npixels"] = [32, 8, 16]
            shotgun_params_kwargs["allsky"] = [False]

            shotgun_params_kwargs["d_eps"] = [1.2, 4.0]
            shotgun_params_kwargs["dx_min"] = [150]
            shotgun_params_kwargs["dx_max"] = [300]
            shotgun_params_kwargs["dy_min"] = [0]
            shotgun_params_kwargs["dy_max"] = [50]
            shotgun_params_kwargs["bins"] = int(500)
            shotgun_params_kwargs["hist_range"] = [(0,500)]

            shotgun_params_kwargs["position_hint"] = [reducedfit.position_hint]
            shotgun_params_kwargs["size_hint"] = [reducedfit.size_hint]

            return super().build_wcs(reducedfit)
        elif reducedfit.obsmode == OBSMODES.POLARIMETRY:
            if ((src_header_obj := reducedfit.rawfit.header_objecthint) is None):
                raise Exception(f"I dont know which object is this supposed to be.")
            
        
    @classmethod
    def get_header_hintcoord(cls, rawfit):
        """ Overriden for DIPOL

        As of 2023-10-23, DIPOL does not inclide RA and DEC in the header, RA and DEC will be derived from the object name.
        """     
        
        return rawfit.header_objecthint.coord