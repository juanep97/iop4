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
    
    @classmethod
    def from_db(cls, db, *args, **kwargs):
        instance = super(ReducedFit, cls).from_db(db, *args, **kwargs)
        instance.auto_merge_to_db=True,
        return instance

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
            self.sources_in_field.set(sources_in_field)
                
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

        fits.writeto(self.filepath, data_new, header=self.header, overwrite=True)

    
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
        return Telescope.by_name(self.telescope).get_astrometry_position_hint(self.rawfit, allsky=allsky,  n_field_width=n_field_width)
    
    def get_astrometry_size_hint(self):
        return Telescope.by_name(self.telescope).get_astrometry_size_hint(self.rawfit)


    # REDUCTION METHODS

    def compute_aperture_photometry(self):
        from .aperphotresult import AperPhotResult
        from iop4lib.utils.sourcedetection import get_bkg, get_segmentation
        from photutils.utils import circular_footprint
        from photutils.aperture import CircularAperture, aperture_photometry
        from photutils.utils import calc_total_error

        if self.mdata.shape[0] == 1024:
            bkg_box_size = 128
        elif self.mdata.shape[0] == 2048:
            bkg_box_size = 256
        elif self.mdata.shape[0] == 800:
            bkg_box_size = 100

        bkg = get_bkg(self.mdata, filter_size=1, box_size=bkg_box_size)
        img_bkg_sub = self.mdata - bkg.background

        if np.sum(self.mdata <= 0.0) >= 1:
            logger.debug(f"{self}: {np.sum(self.mdata <= 0.0)} px < 0  ({math.sqrt(np.sum(self.mdata <= 0.0)):0f} px2) in IMAGE.")

        if np.sum(img_bkg_sub <= 0.0) >= 1:
            logger.debug(f"{self}: {np.sum(img_bkg_sub <= 0.0)} px < 0 ({math.sqrt(np.sum(img_bkg_sub <= 0.0)):.0f} px2) in BKG-SUBSTRACTED IMG. Check if the bkg-substraction method, I'm going to try to mask sources...")
            seg_threshold = 3.0 * bkg.background_rms # safer to ensure they are sources
            segment_map, convolved_data = get_segmentation(img_bkg_sub, threshold=seg_threshold, fwhm=1, kernel_size=None, npixels=16, deblend=True)
            mask = segment_map.make_source_mask(footprint=circular_footprint(radius=6))
            bkg = get_bkg(self.mdata, filter_size=1, box_size=bkg_box_size, mask=mask)
            img_bkg_sub = self.mdata - bkg.background
        
        if np.sum(img_bkg_sub <= 0.0) >= 1:
            logger.debug(f"{self}: {np.sum(img_bkg_sub <= 0.0)} px < 0 ({math.sqrt(np.sum(img_bkg_sub <= 0.0)):.0f} px2) in BKG-SUBSTRACTED IMG, after masking.")


        effective_gain = Telescope.by_name(self.epoch.telescope).gain_e_adu
        error = calc_total_error(img_bkg_sub, bkg.background_rms, effective_gain)

        for astrosource in self.sources_in_field.all():
            for pairs, wcs in (('O', self.wcs1), ('E', self.wcs2)) if self.with_pairs else (('O',self.wcs),):

                aperpix = astrosource.get_aperpix()

                ap = CircularAperture(astrosource.coord.to_pixel(wcs), r=aperpix)

                bkg_aperphot_tb = aperture_photometry(bkg.background, ap, error=error)
                bkg_flux_counts = bkg_aperphot_tb['aperture_sum'][0]
                bkg_flux_counts_err = bkg_aperphot_tb['aperture_sum_err'][0]

                aperphot_tb = aperture_photometry(img_bkg_sub, ap, error=error)
                flux_counts = aperphot_tb['aperture_sum'][0]
                flux_counts_err = aperphot_tb['aperture_sum_err'][0]

                AperPhotResult.create(reducedfit=self, 
                                      astrosource=astrosource, 
                                      aperpix=aperpix, 
                                      pairs=pairs, 
                                      bkg_flux_counts=bkg_flux_counts, bkg_flux_counts_err=bkg_flux_counts_err,
                                      flux_counts=flux_counts, flux_counts_err=flux_counts_err)
    
    

    def compute_relative_photometry(self):

        if self.obsmode != OBSMODES.PHOTOMETRY:
            raise Exception(f"{self}: this method is only for plain photometry images.")

        # 1. Compute all aperture photometries

        logger.debug(f"{self}: computing aperture photometries for {self}.")

        self.compute_aperture_photometry()

        # 2. Compute relative polarimetry for each source (uses the computed aperture photometries)

        logger.debug(f"{self}: computing relative photometry.")

        # 2. Compute the flux in counts and the instrumental magnitude
        
        photopolresult_L = list()
        
        for astrosource in self.sources_in_field.all():
            result = PhotoPolResult.create(reducedfits=[self], astrosource=astrosource, reduction=REDUCTIONMETHODS.RELPHOT) # aperpix auto

            aperphotresult = AperPhotResult.objects.get(reducedfit=self, astrosource=astrosource)

            result.bkg_flux_counts = aperphotresult.bkg_flux_counts
            result.bkg_flux_counts_err = aperphotresult.bkg_flux_counts_err
            result.flux_counts = aperphotresult.flux_counts
            result.flux_counts_err = aperphotresult.flux_counts_err

            # logger.debug(f"{self}: {result.flux_counts=}")

            if result.flux_counts <= 0.0:
                logger.warning(f"{self}: negative flux counts encountered while relative photometry for {astrosource=} ??!! They will be nans, but maybe we should look into this...")

            result.mag_inst = -2.5 * np.log10(result.flux_counts) # np.nan if result.flux_counts <= 0.0
            result.mag_inst_err = math.fabs(2.5 / math.log(10) / result.flux_counts * result.flux_counts_err)

            # if the source is a calibrator, compute also the zero point
            if result.astrosource.srctype == SRCTYPES.CALIBRATOR:
                result.mag_known = getattr(result.astrosource, f"mag_{self.band}")
                result.mag_known_err = getattr(result.astrosource, f"mag_{self.band}_err", None) or 0.0

                if result.mag_known is None:
                    logger.warning(f"Relative Photometry over {self}: calibrator {result.astrosource} has no magnitude for band {self.band}.")
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
            logger.error(f"{self}: can not perform relative photometry without any calibrators for this reduced fit. Deleting results.")
            [result.delete() for result in self.photopolresults.all()]
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




    @classmethod
    def compute_relative_polarimetry_caha(cls, polarimetry_group):
        """ Computes the relative polarimetry for a polarimetry group for CAHA T220 observations."""
        
        # Perform some checks on the group

        ## get the band of the group

        bands = [reducedfit.band for reducedfit in polarimetry_group]

        if len(set(bands)) == 1:
            band = bands[0]
        else: # should not happens
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
        rot_angles_required = {0.0, 22.48, 44.98, 67.48}

        if not rot_angles_available.issubset(rot_angles_required):
            logger.warning(f"Rotation angles missing: {rot_angles_required - rot_angles_available}")

        # 1. Compute all aperture photometries

        logger.debug(f"Computing aperture photometries for the {len(polarimetry_group)} reducedfits in the group.")

        for reducedfit in polarimetry_group:
            reducedfit.compute_aperture_photometry()

        # 2. Compute relative polarimetry for each source (uses the computed aperture photometries)

        logger.debug("Computing relative polarimetry.")

        photopolresult_L = list()

        for astrosource in group_sources:
            logger.debug(f"Computing relative polarimetry for {astrosource}.")

            # if any rotator angle is missing, uses the extraordinaty of other angles

            obs_0 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, reducedfit__rotangle=0.0).exists()
            obs_22 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, reducedfit__rotangle=22.48).exists()
            obs_45 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, reducedfit__rotangle=44.98).exists()
            obs_67 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, reducedfit__rotangle=67.48).exists()

            if not obs_0 or not obs_22 or not obs_45 or not obs_67:
                logger.warning(f"missing rotangles for {astrosource}")

            qs_O_0 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O" if obs_0 else "E", reducedfit__rotangle=0.0 if obs_0 else 44.98)
            flux_O_0, flux_O_0_err = qs_O_0.values_list("flux_counts", "flux_counts_err").last()

            qs_O_22 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O" if obs_22 else "E", reducedfit__rotangle=22.48 if obs_0 else 67.48)
            flux_O_22, flux_O_22_err = qs_O_22.values_list("flux_counts", "flux_counts_err").last()

            qs_O_45 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O" if obs_45 else "E", reducedfit__rotangle=44.98 if obs_45 else 0.0)
            flux_O_45, flux_O_45_err = qs_O_45.values_list("flux_counts", "flux_counts_err").last()

            qs_O_67 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O" if obs_67 else "E", reducedfit__rotangle=67.48 if obs_67 else 22.48)
            flux_O_67, flux_O_67_err = qs_O_67.values_list("flux_counts", "flux_counts_err").last()

            qs_E_0 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="E" if obs_0 else "O", reducedfit__rotangle=0.0 if obs_0 else 44.98)
            flux_E_0, flux_E_0_err = qs_E_0.values_list("flux_counts", "flux_counts_err").last()

            qs_E_22 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="E" if obs_22 else "O", reducedfit__rotangle=22.48 if obs_22 else 67.48)
            flux_E_22, flux_E_22_err = qs_E_22.values_list("flux_counts", "flux_counts_err").last()

            qs_E_45 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="E" if obs_45 else "O", reducedfit__rotangle=44.98 if obs_45 else 0.0)
            flux_E_45, flux_E_45_err = qs_E_45.values_list("flux_counts", "flux_counts_err").last()

            qs_E_67 = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="E" if obs_67 else "O", reducedfit__rotangle=67.48 if obs_67 else 22.48)
            flux_E_67, flux_E_67_err = qs_E_67.values_list("flux_counts", "flux_counts_err").last()

            fluxes_O = np.array([flux_O_0, flux_O_22, flux_O_45, flux_O_67])
            fluxes_E = np.array([flux_E_0, flux_E_22, flux_E_45, flux_E_67])

            if np.any(fluxes_O <= 0) or np.any(fluxes_E <= 0):
                logger.warning(f"{astrosource}: fluxes <= 0 !!")
                logger.debug(f"Fluxes_O: {fluxes_O}")
                logger.debug(f"Fluxes_E: {fluxes_E}")

            fluxes = (fluxes_O + fluxes_E) /2.
            flux_mean = fluxes.mean()
            flux_std = fluxes.std()

            RQ = np.sqrt((flux_O_0 / flux_E_0) / (flux_O_45 / flux_E_45))
            dRQ = RQ * np.sqrt((flux_O_0_err / flux_O_0) ** 2 + (flux_E_0_err / flux_E_0) ** 2 + (flux_O_45_err / flux_O_45) ** 2 + (flux_E_45_err / flux_E_45) ** 2)

            RU = np.sqrt((flux_O_0 / flux_E_22) / (flux_O_67 / flux_E_67))
            dRU = RU * np.sqrt((flux_O_22_err / flux_O_22_err) ** 2 + (flux_E_22 / flux_E_22_err) ** 2 + (flux_O_67_err / flux_O_67) ** 2 + (flux_E_67_err / flux_E_67) ** 2)
        
            Q_I = (RQ - 1) / (RQ + 1)
            dQ_I = Q_I * np.sqrt(2 * (dRQ / RQ) ** 2)
            U_I = (RU - 1) / (RU + 1)
            dU_I = U_I * np.sqrt(2 * (dRU / RU) ** 2)

            P = np.sqrt(Q_I ** 2 + U_I ** 2)
            dP = P * np.sqrt((dRQ / RQ) ** 2 + (dRU / RU) ** 2) / 2

            Theta_0 = 0
        
            if Q_I >= 0:
                Theta_0 = math.pi 
                if U_I > 0:
                    Theta_0 = -1 * math.pi
                # if Q_I < 0:
                #     Theta_0 = math.pi / 2
                
            Theta = 0.5 * math.degrees(math.atan(U_I / Q_I) + Theta_0)
            dTheta = dP / P * 28.6

            # compute instrumental magnitude

            if flux_mean <= 0.0:
                logger.warning(f"{polarimetry_group=}: negative flux mean encountered while relative polarimetry for {astrosource=} ??!! It will be nan, but maybe we should look into this...")

            mag_inst = -2.5 * np.log10(flux_mean)
            mag_inst_err = math.fabs(2.5 / math.log(10) * flux_std / flux_mean)

            # if the source is a calibrator, compute also the zero point

            if astrosource.srctype == SRCTYPES.CALIBRATOR:
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
                    
            result = PhotoPolResult.create(reducedfits=polarimetry_group, 
                                                           astrosource=astrosource, 
                                                           reduction=REDUCTIONMETHODS.RELPOL, 
                                                           mag_inst=mag_inst, mag_inst_err=mag_inst_err, mag_zp=mag_zp, mag_zp_err=mag_zp_err,
                                                           flux_counts=flux_mean, p=P, p_err=dP, chi=Theta, chi_err=dTheta)
            
            photopolresult_L.append(result)


        # 3. Get average zero point from zp of all calibrators in the group

        calib_mag_zp_array = np.array([result.mag_zp or np.nan for result in photopolresult_L if result.astrosource.srctype == SRCTYPES.CALIBRATOR]) # else it fills with None also and the dtype becomes object
        calib_mag_zp_array = calib_mag_zp_array[~np.isnan(calib_mag_zp_array)]

        calib_mag_zp_array_err = np.array([result.mag_zp_err or np.nan for result in photopolresult_L if result.astrosource.srctype == SRCTYPES.CALIBRATOR])
        calib_mag_zp_array_err = calib_mag_zp_array_err[~np.isnan(calib_mag_zp_array_err)]

        if len(calib_mag_zp_array) == 0:
            logger.error(f"Can not compute magnitude during relative photo-polarimetry without any calibrators for this reduced fit.")

        zp_avg = np.nanmean(calib_mag_zp_array)
        zp_std = np.nanstd(calib_mag_zp_array)

        zp_err = np.sqrt(np.nansum(calib_mag_zp_array_err ** 2)) / len(calib_mag_zp_array_err)
        zp_err = math.sqrt(zp_err ** 2 + zp_std ** 2)

        # 4. Compute the calibrated magnitudes for non-calibrators in the group using the averaged zero point

        for result in photopolresult_L:

            if result.astrosource.srctype == SRCTYPES.CALIBRATOR:
                continue

            result.mag_zp = zp_avg
            result.mag_zp_err = zp_err
        
            result.mag = result.mag_inst + zp_avg
            result.mag_err = math.sqrt(result.mag_inst_err ** 2 + zp_err ** 2)

            result.save()

        # 5. Save results
        for result in photopolresult_L:
            result.save()




    @classmethod
    def compute_relative_polarimetry_osnt090(cls, polarimetry_group):
        """ Computes the relative polarimetry for a polarimetry group for OSNT090 observations."""

        logger.debug("Computing OSN-T090 relative polarimetry for group: %s", "".join(map(str,polarimetry_group)))

        #values for T090 TODO: update them

        qoff = 0.0579
        uoff = 0.0583
        dqoff = 0.003
        duoff = 0.0023
        Phi = math.radians(-18)
        dPhi = math.radians(0.001)


        # Perform some checks on the group

        ## get the band of the group

        bands = [reducedfit.band for reducedfit in polarimetry_group]

        if len(set(bands)) == 1:
            band = bands[0]
        else: # should not happens
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
        rot_angles_required = {0.0, 45.0, 90.0, -45.0}

        if not rot_angles_available.issubset(rot_angles_required):
            logger.warning(f"Rotation angles missing: {rot_angles_required - rot_angles_available}")

        # 1. Compute all aperture photometries

        logger.debug(f"Computing aperture photometries for the {len(polarimetry_group)} reducedfits in the group.")

        for reducedfit in polarimetry_group:
            reducedfit.compute_aperture_photometry()

        # 2. Compute relative polarimetry for each source (uses the computed aperture photometries)

        logger.debug("Computing relative polarimetry.")

        photopolresult_L = list()

        for astrosource in group_sources:
            
            flux_0 = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", reducedfit__rotangle=0.0).flux_counts
            flux_0_err = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", reducedfit__rotangle=0.0).flux_counts_err

            flux_45 = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", reducedfit__rotangle=45.0).flux_counts
            flux_45_err = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", reducedfit__rotangle=45.0).flux_counts_err

            flux_90 = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", reducedfit__rotangle=90.0).flux_counts
            flux_90_err = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", reducedfit__rotangle=90.0).flux_counts_err

            flux_n45 = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", reducedfit__rotangle=-45.0).flux_counts
            flux_n45_err = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", reducedfit__rotangle=-45.0).flux_counts_err

            # from IOP3 polarimetry_osn() :

            fluxes = np.array([flux_0, flux_45, flux_90, flux_n45])
            flux_mean = fluxes.mean()
            flux_std = fluxes.std()
            
            qraw = (flux_0 - flux_90) / (flux_0 + flux_90)
            uraw = (flux_45 - flux_n45) / (flux_45 + flux_n45)
            
            #Applying error propagation...
            
            dqraw = qraw * math.sqrt(((flux_0_err**2+flux_90_err**2)/(flux_0+flux_90)**2)+(((flux_0_err**2+flux_90_err**2))/(flux_0-flux_90)**2))
            duraw = uraw * math.sqrt(((flux_45_err**2+flux_n45_err**2)/(flux_45+flux_n45)**2)+(((flux_45_err**2+flux_n45_err**2))/(flux_45-flux_n45)**2))

            qc = qraw - qoff
            uc = uraw - uoff

            dqc = math.sqrt(dqraw**2 + dqoff**2) 
            duc = math.sqrt(duraw**2 + duoff**2)

            q = qc*math.cos(2*Phi) - uc*math.sin(2*Phi)
            u = qc*math.sin(2*Phi) + uc*math.cos(2*Phi)
            
            #Comment these to get instrumental polarization
            dqa = qc*math.cos(2*Phi) * math.sqrt((dqc/qc)**2+((2*dPhi*math.sin(2*Phi))/(math.cos(2*Phi)))**2) 
            dqb = uc*math.sin(2*Phi) * math.sqrt((duc/uc)**2+((2*dPhi*math.cos(2*Phi))/(math.sin(2*Phi)))**2)
            dua = qc*math.sin(2*Phi) * math.sqrt((dqc/qc)**2+((2*dPhi*math.cos(2*Phi))/(math.sin(2*Phi)))**2) 
            dub = uc*math.cos(2*Phi) * math.sqrt((duc/uc)**2+((2*dPhi*math.sin(2*Phi))/(math.cos(2*Phi)))**2)
            
            #For instrumental polarization
            #dqa=0
            #dua=0
            #dqb=0
            #dub=0
            
            dq = np.sqrt(dqa**2+dqb**2)
            du = np.sqrt(dua**2+dub**2)
            
            P = math.sqrt(q**2 + u**2)
            dP = P * (1/(q**2+u**2)) * math.sqrt((q*dq)**2+(u*du)**2)
            
            Theta_0 = 0
            if q >=0:
                Theta_0 = math.pi
                if u > 0:
                    Theta_0 = -1 * math.pi

            #    Theta_0 = 0
            #    if q > 0:
            #        if u >= 0:
            #            Theta_0 = 0
            #        if u < 0:
            #            Theta_0 = math.pi / 2
            #    elif q < 0:
            #        Theta_0 = math.pi / 4

            Theta_0 = 0
            Theta = (1/2) * math.degrees(math.atan2(u,q) + Theta_0)
            dTheta = dP/P * 28.6

            # same as for caha now: 
            
            # compute instrumental magnitude

            if flux_mean <= 0.0:
                logger.warning(f"{polarimetry_group=}: negative flux mean encountered while relative polarimetry for {astrosource=} ??!! It will be nan, but maybe we should look into this...")

            mag_inst = -2.5 * np.log10(flux_mean)
            mag_inst_err = math.fabs(2.5 / math.log(10) * flux_std / flux_mean)

            # if the source is a calibrator, compute also the zero point

            if astrosource.srctype == SRCTYPES.CALIBRATOR:
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
                    
            result = PhotoPolResult.create(reducedfits=polarimetry_group, 
                                           astrosource=astrosource, 
                                           reduction=REDUCTIONMETHODS.RELPOL, 
                                           mag_inst=mag_inst, mag_inst_err=mag_inst_err, mag_zp=mag_zp, mag_zp_err=mag_zp_err,
                                           flux_counts=flux_mean, p=P, p_err=dP, chi=Theta, chi_err=dTheta)
            
            photopolresult_L.append(result)


        # 3. Get average zero point from zp of all calibrators in the group

        calib_mag_zp_array = np.array([result.mag_zp or np.nan for result in photopolresult_L if result.astrosource.srctype == SRCTYPES.CALIBRATOR]) # else it fills with None also and the dtype becomes object
        calib_mag_zp_array = calib_mag_zp_array[~np.isnan(calib_mag_zp_array)]

        calib_mag_zp_array_err = np.array([result.mag_zp_err or np.nan for result in photopolresult_L if result.astrosource.srctype == SRCTYPES.CALIBRATOR])
        calib_mag_zp_array_err = calib_mag_zp_array_err[~np.isnan(calib_mag_zp_array_err)]

        if len(calib_mag_zp_array) == 0:
            logger.error(f"Can not compute magnitude during relative photo-polarimetry without any calibrators for this reduced fit.")

        zp_avg = np.nanmean(calib_mag_zp_array)
        zp_std = np.nanstd(calib_mag_zp_array)

        zp_err = np.sqrt(np.nansum(calib_mag_zp_array_err ** 2)) / len(calib_mag_zp_array_err)
        zp_err = math.sqrt(zp_err ** 2 + zp_std ** 2)

        # 4. Compute the calibrated magnitudes for non-calibrators in the group using the averaged zero point

        for result in photopolresult_L:

            if result.astrosource.srctype == SRCTYPES.CALIBRATOR:
                continue

            result.mag_zp = zp_avg
            result.mag_zp_err = zp_err
        
            result.mag = result.mag_inst + zp_avg
            result.mag_err = math.sqrt(result.mag_inst_err ** 2 + zp_err ** 2)

            result.save()

        # 5. Save results
        for result in photopolresult_L:
            result.save()

