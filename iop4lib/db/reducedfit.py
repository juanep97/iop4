# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.db import models
from django.forms.models import model_to_dict

# other imports
import os
import math
import numpy as np
import astropy.units as u
from astropy.wcs import WCS
import astropy.io.fits as fits

# iop4lib imports
from iop4lib.telescopes import Telescope
from iop4lib.instruments import Instrument
from iop4lib.utils.filedproperty import FiledProperty
from iop4lib.enums import *
from .rawfit import RawFit
from .astrosource import AstroSource
from .photopolresult import PhotoPolResult
from .aperphotresult import AperPhotResult

# logging
import logging
logger = logging.getLogger(__name__)

    
class ReducedFit(RawFit):
    """
    Reduced fits (bias and flat corrected, with wcs coordinates, etc, source list, etc).
    """

    # Database fields and information.

    # identifier
    rawfit = models.OneToOneField('RawFit', on_delete=models.CASCADE, related_name='reduced', parent_link=True, help_text="RawFit of this ReducedFit.")

    # ReducedFit specific fields
    masterbias = models.ForeignKey('MasterBias', null=True, on_delete=models.CASCADE, related_name='reduced', help_text="MasterBias to be used for the reduction.")
    masterflat = models.ForeignKey('MasterFlat', null=True, on_delete=models.CASCADE, related_name='reduced', help_text="MasterFlat to be used for the reduction.")
    sources_in_field = models.ManyToManyField('AstroSource', related_name='in_reducedfits', blank=True, help_text="Sources in the field of this FITS.")
    modified = models.DateTimeField(auto_now=True, help_text="Last time this entry was modified.")

    class Meta(RawFit.Meta):
        app_label = "iop4api"
        verbose_name = "ReducedFit"
        verbose_name_plural = "Reduced FITS files"

    # Properties

    @property
    def filepath(self):
       """ Returns the path to the file.
       The reduced FITS files are stored in the .filedpropdir directory (the same .filedpropdir as the corresponding RawFit).
       """
       return os.path.join(self.filedpropdir, self.filename)
    
    # Filed properties
    astrometry_info = FiledProperty()

    # Constructors

    @classmethod
    def create(cls, rawfit, 
               masterbias=None,
               masterflat=None,
               auto_build=False,
               force_rebuild=False,
               auto_merge_to_db=True):
        """Creates a new instance of ReducedFit from a RawFit, updates existing entry if it exists.

        Parameters
        ----------
        rawfit : RawFit
            The RawFit to be reduced.
        masterbias : MasterBias, optional
            The MasterBias to be used for the reduction. If None, an appropiate MasterBias will be found.
        masterflat : MasterFlat, optional
            The MasterFlat to be used for the reduction. If None, an appropiate MasterFlat will be found.

        Other Parameters
        ----------------
        auto_build : bool, optional
            If auto_build is True, the file will be built if it does not exist.
        force_rebuild : bool, optional
            If force_rebuild is True, the file will be built regardless of whether it exists or not; this takes 
            precedence over auto_build.

        Returns
        -------
        None

        Notes
        -----
            See ReducedFit.build_file for more information about how the file is build.
        """
        from iop4lib.db import RawFit

        if rawfit is None:
            raise ValueError("rawfit must be provided.")
        
        if rawfit.imgtype != IMGTYPES.LIGHT:
            raise ValueError("Only RawFits of type LIGHT can be reduced.")
        
        if (reduced := ReducedFit.objects.filter(rawfit=rawfit).first()) is None:
            logger.debug(f"Creating DB entry for reduced fit {rawfit.fileloc}")     
            # get fields from rawfit, else they get overwritten
            # TODO: check if there is a better way to do this
            #field_dict = model_to_dict(rawfit)
            field_dict = RawFit.objects.filter(id=rawfit.id).values().get()
            logger.debug(f"Trying to create ReducedFit from RawFit, {field_dict=}. Check code if you encounter any problem. Might give an error if some foreign field is stored in rawfit. TODO: might be better to use .save_base()?")
            reduced = cls(rawfit=rawfit, **field_dict)
            reduced.save()
        else:
            logger.debug(f"DB entry of ReducedFit for {rawfit.fileloc} already exists, it will be used instead.")

        # instance only attributes
        reduced.auto_merge_to_db = auto_merge_to_db

        # associate a masterbias to this reducedfit

        if masterbias is not None:
            reduced.masterbias = masterbias
        else:
            if (mb := rawfit.request_masterbias()) is not None:
                reduced.masterbias = mb
            else:
                logger.warning(f"{reduced}: MasterBias in this epoch could not be found, attemptying adjacent epochs.")
                if (mb := rawfit.request_masterbias(other_epochs=True)) is not None:
                    reduced.masterbias = mb
                else:
                    logger.error(f"{reduced}: Could not find any MasterBias, not even in adjacent epochs.")
                    reduced.set_flag(ReducedFit.FLAGS.ERROR)

        # associate a masterflat to this reducedfit

        if masterflat is not None:
            reduced.masterflat = masterflat
        else:
            if (mf := rawfit.request_masterflat()) is not None:
                reduced.masterflat = mf
            else:
                logger.warning(f"{reduced}: MasterFlat in this epoch could not be found, attemptying adjacent epochs.")
                if (mf := rawfit.request_masterflat(other_epochs=True)) is not None:
                    reduced.masterflat = mf
                else:
                    logger.error(f"{reduced}: Could not find any MasterFlat, not even in adjacent epochs.")
                    reduced.set_flag(ReducedFit.FLAGS.ERROR)

        # build file

        if (not reduced.fileexists and auto_build) or force_rebuild:
            logger.info(f"Building file")
            reduced.build_file()

        # merge to db

        if auto_merge_to_db:
            reduced.save()

        return reduced
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_merge_to_db = True
    

    # Calibration methods

    def build_file(self):
        """ Builds the ReducedFit FITS file.

        Notes
        -----
        The file is built by:
        - applying masterbias..
        - applying masterflat.
        - try to astrometerically calibrate the reduced fit, giving it a WCS.
        - find the catalog sources in the field.
        """

        logger.debug(f"{self}: building file")

        self.unset_flag(ReducedFit.FLAGS.BUILT_REDUCED)

        logger.debug(f"{self}: applying masterbias {self.masterbias}")
        self.apply_masterbias()

        logger.debug(f"{self}: applying masterflat {self.masterflat}")
        self.apply_masterflat()
        
        logger.debug(f"{self}: performing astrometric calibration")

        try:
            self.astrometric_calibration()
        except Exception as e:
            logger.error(f"{self}: could not perform astrometric calibration on {self}: {e}")
            self.set_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY)
            if self.auto_merge_to_db:
                self.save()
            raise e
        else:
            logger.debug(f"{self}: astrometric calibration was successful.")
            self.unset_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY)

            logger.debug(f"{self}: searching for sources in field...")
            sources_in_field = AstroSource.get_sources_in_field(fit=self)
            
            logger.debug(f"{self}: found {len(sources_in_field)} sources in field.")
            self.sources_in_field.set(sources_in_field, clear=True)
                
            self.set_flag(ReducedFit.FLAGS.BUILT_REDUCED)

        if self.auto_merge_to_db:
            self.save()


    def apply_masterbias(self):
        """ Applies the masterbias to the rawfit.

        It starts from the RawFit FITS file. This creates the ReducedFit file for the first time.
        """

        import astropy.io.fits as fits

        rf_data = fits.getdata(self.rawfit.filepath)
        mb_data = fits.getdata(self.masterbias.filepath)

        data_new = rf_data - mb_data

        if not os.path.exists(os.path.dirname(self.filepath)):
            logger.debug(f"{self}: creating directory {os.path.dirname(self.filepath)}")
            os.makedirs(os.path.dirname(self.filepath))

        # header_new = self.rawfit.header

        # # remove blank keyword
        # # unlinke the rawfit, the reduced fit now contains float values, so the blank keyword is non standard
        # # and will cause warnings, we remove it from the rawfit header.
        # if 'BLANK' in header_new:
        #     del header_new['BLANK']

        # better create a new header beacuse the previous one had bad keywords for wcs, they will give problems

        header_new = fits.Header()

        fits.writeto(self.filepath, data_new, header=header_new, overwrite=True)

    def apply_masterflat(self):
        """ Applies the masterflat to the rawfit.
        
        Notes
        -----
        #no longer: - This normalizes by exposure time.
        - self.apply_masterbias() must have been called before, as it creates the reduced FIT file.
        """

        import numpy as np
        import astropy.io.fits as fits

        data = fits.getdata(self.filepath)
        mf_data = fits.getdata(self.masterflat.filepath)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #data_new = (data/self.rawfit.exptime) / (mf_data)
            data_new = (data) / (mf_data)


        # Reduce precission

        # Assuming data_new is your floating-point data array
        data_max = data_new.max()
        data_min = data_new.min()

        # Calculate the scaling factor
        output_dtype = np.int16 # or np.int32

        idtype_max = np.iinfo(output_dtype).max
        idtype_min = np.iinfo(output_dtype).min

        # either:

        # Apply the scaling factor and offset to the data
        m = (idtype_max - idtype_min) / (data_max - data_min)
        n = - data_min * (idtype_max-idtype_min) / (data_max - data_min) + idtype_min
        
        bscale = 1/m
        bzero = -n/m

        # with fits.open(self.filepath, "update") as hdul:
        #     hdul[0].data = ( m*data_new + n ).astype(output_dtype)
        #     hdul[0].header.update({'BSCALE': bscale, 'BZERO': bzero})

        # or:

        hdu = fits.PrimaryHDU(data_new.copy(), header=fits.Header())
        hdu.scale(np.dtype(output_dtype).name, bscale=bscale, bzero=bzero)
        hdul = fits.HDUList([hdu])
        hdul.writeto(self.filepath, overwrite=True)

        # other methods like simple that access the data  fits.writeto(data, header=header) will mess again the type

        logger.warning(f"{data_new.dtype=}, {self.header['BSCALE']=}, {self.header['BZERO']=}")
        logger.warning(f"{self.data.dtype=}, {self.header['BSCALE']=}, {self.header['BZERO']=}")
        diff = np.abs(data_new - self.data)

        # logger.warning("Max value of the original fit: %f", data_new.mean())
        # logger.warning("Max value of the reduced fit: %f", self.data.mean())
        # logger.warning("Min value of the original fit: %f", data_new.min())
        # logger.warning("Min value of the reduced fit: %f", self.data.min())
        logger.error("Mean value of the original fit: %f", data_new.mean())
        logger.error("Mean value of the reduced fit: %f", self.data.mean())

        # logger.warning("Max difference between original and reduced fit: %f", diff.max())
        # logger.warning("Min difference between original and reduced fit: %f", diff.min())
        logger.error("Mean difference between original and reduced fit: %f", diff.mean())

        # logger.warning(f"Max difference (relative) between original and reduced fit: {100*(diff/data_new).max():.2f} %")
        # logger.warning(f"Min difference (relative) between original and reduced fit: {100*(diff/data_new).min():.2f} %")
        logger.error(f"Mean difference (relative) between original and reduced fit: {100*(diff/data_new).mean():.2f} %")

        if 'BSCALE' not in fits.getheader(self.filepath):
            raise Exception(f"Could not write BSCALE to {self.filepath}")
        
        logger.warning(f"Reduced {self.id} is {os.path.getsize(self.filepath)/1024/1024:.1f} MB vs {os.path.getsize(self.rawfit.filepath)/1024/1024:.1f} MB of rawfit")


    @property
    def with_pairs(self):
        """ Indicates whether both ordinary and extraordinary sources are present 
        in the file. At the moment, this happens only for CAFOS polarimetry
        """
        return (self.rawfit.instrument == INSTRUMENTS.CAFOS and self.rawfit.obsmode == OBSMODES.POLARIMETRY)
        
    def astrometric_calibration(self):
        """ Performs astrometric calibration on the reduced fit, giving it the appropriate WCS.

        If the are both ordinary and extraordinary sources in the field, one WCS will be built for each,
        and the will be saved in the first and second extensions of the FITS file.
        """
        from iop4lib.utils.astrometry import build_wcs

        build_wcs_result = build_wcs(self)

        if build_wcs_result['success']:

            logger.debug(f"{self}: saving WCSs to FITS header.")

            wcs1 = build_wcs_result['wcslist'][0]

            header = fits.Header()

            header.update(wcs1.to_header(relax=True, key="A"))

            if self.with_pairs:
                wcs2 = build_wcs_result['wcslist'][1]
                header.update(wcs2.to_header(relax=True, key="B"))

            # if available, save also some info about the astrometry solution
            if 'bm' in build_wcs_result['info']:
                bm = build_wcs_result['info']['bm']
                # adding HIERARCH avoids a warning, they can be accessed without HIERARCH
                header['HIERARCH AS_ARCSEC_PER_PIX'] = bm.scale_arcsec_per_pixel
                header['HIERARCH AS_CENTER_RA_DEG'] = bm.center_ra_deg
                header['HIERARCH AS_CENTER_DEC_DEG'] = bm.center_dec_deg

            with fits.open(self.filepath, 'update') as hdul:
                hdul[0].header.update(header)

        else:
            raise Exception(f"Could not perform astrometric calibration on {self}: {build_wcs_result=}")

    @property
    def wcs(self):
        """ Returns the WCS of the reduced fit. """
        return WCS(self.header, key="A")
    
    @property
    def wcs1(self):
        """ Returns the WCS of the first extension of the reduced fit. """
        return WCS(self.header, key="A")
    
    @property
    def wcs2(self):
        """ Returns the WCS of the second extension of the reduced fit. """
        return WCS(self.header, key="B")

    @property
    def pixscales(self):
        """ Pixel scales of the reduced fit as quantities (deg/pix) (it returns both the x and y scales)."""
        pix_scales_deg_pix = np.sqrt(np.sum(self.wcs.pixel_scale_matrix**2, axis=0))
        return pix_scales_deg_pix * u.Unit("deg / pix")
    
    @property
    def pixscale(self):
        """ Pixel scale of the reduced fit, in units of deg/pix (it returns the mean of x and y scales so it is strictly correct only for square pixels). """
        return  np.mean(self.pixscales)
    
    @property
    def centercoord(self):
        return self.wcs.pixel_to_world(self.mdata.shape[0]//2, self.mdata.shape[1]//2)

    @property
    def pixscale_equiv(self):
        """ Returns an equivalencies between pixels and angles for this reduced fit. """
        return u.pixel_scale(self.pixscale)
    
    @property
    def header_hintcoord(self):
        return self.rawfit.header_hintcoord
    
    @property
    def header_objecthint(self):
        return self.rawfit.header_objecthint
    
    def get_astrometry_position_hint(self, allsky=False, n_field_width=1.5):
        return Instrument.by_name(self.instrument).get_astrometry_position_hint(self.rawfit, allsky=allsky,  n_field_width=n_field_width)
    
    def get_astrometry_size_hint(self):
        return Instrument.by_name(self.instrument).get_astrometry_size_hint(self.rawfit)


    # REDUCTION METHODS

    ## Delegated to telescopes
    
    def compute_aperture_photometry(self, *args, **kwargs):
        return Instrument.by_name(self.instrument).compute_aperture_photometry(self, *args, **kwargs)

    def compute_relative_photometry(self, *args, **kwargs):
        return Instrument.by_name(self.instrument).compute_relative_photometry(self, *args, **kwargs)
    
    @classmethod
    def compute_relative_polarimetry(cls, polarimetry_group, *args, **kwargs):
        
        if not all([redf.telescope == polarimetry_group[0].telescope for redf in polarimetry_group]):
            raise Exception("All reduced fits in a polarimetry group must be from the same telescope")
        
        if not all([redf.instrument == polarimetry_group[0].instrument for redf in polarimetry_group]):
            raise Exception("All reduced fits in a polarimetry group must be from the same instrument")
        
        return Instrument.by_name(polarimetry_group[0].telescope).compute_relative_polarimetry(polarimetry_group, *args, **kwargs)