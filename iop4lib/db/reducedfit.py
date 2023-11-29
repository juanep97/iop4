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
    masterdark = models.ForeignKey('MasterDark', null=True, on_delete=models.CASCADE, related_name='reduced', help_text="MasterDark to be used for the reduction.")
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
               masterdark=None,
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

        reduced.associate_masters(masterbias=masterbias, masterdark=masterdark, masterflat=masterflat)

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
    

    # REDUCTION METHODS

    ## Delegated to telescopes or instrument classes

    def associate_masters(self, *args, **kwargs):
        return Instrument.by_name(self.instrument).associate_masters(self, *args, **kwargs)

    def apply_masters(self):
        return Instrument.by_name(self.instrument).apply_masters(self)
    
    def build_file(self, **build_wcs_kwargs):
        return Instrument.by_name(self.instrument).build_file(self, **build_wcs_kwargs)

    def astrometric_calibration(self, **build_wcs_kwargs):
        return Instrument.by_name(self.instrument).astrometric_calibration(self, **build_wcs_kwargs)

    @property
    def has_pairs(self):
        """ Indicates whether both ordinary and extraordinary sources are present in the file. """
        return Instrument.by_name(self.instrument).has_pairs(self)
        
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
    def header_hintobject(self):
        return self.rawfit.header_hintobject

    def get_astrometry_position_hint(self, allsky=False, n_field_width=1.5, hintsep=None):
        return Instrument.by_name(self.instrument).get_astrometry_position_hint(self.rawfit, allsky=allsky,  n_field_width=n_field_width, hintsep=hintsep)
    
    def get_astrometry_size_hint(self):
        return Instrument.by_name(self.instrument).get_astrometry_size_hint(self.rawfit)
    
    def compute_aperture_photometry(self, *args, **kwargs):
        """ Delegated to the instrument. """
        return Instrument.by_name(self.instrument).compute_aperture_photometry(self, *args, **kwargs)

    def compute_relative_photometry(self, *args, **kwargs):
        """ Delegated to the instrument. """
        return Instrument.by_name(self.instrument).compute_relative_photometry(self, *args, **kwargs)
    
    @classmethod
    def compute_relative_polarimetry(cls, polarimetry_group, *args, **kwargs):
        """ Delegated to the instrument. """
        
        if not all([redf.telescope == polarimetry_group[0].telescope for redf in polarimetry_group]):
            raise Exception("All reduced fits in a polarimetry group must be from the same telescope")
        
        if not all([redf.instrument == polarimetry_group[0].instrument for redf in polarimetry_group]):
            raise Exception("All reduced fits in a polarimetry group must be from the same instrument")
        
        return Instrument.by_name(polarimetry_group[0].telescope).compute_relative_polarimetry(polarimetry_group, *args, **kwargs)