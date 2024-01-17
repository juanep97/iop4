# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from abc import ABCMeta

# other imports
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
import astrometry
import numpy as np
import math

# iop4lib imports
from iop4lib.enums import *
from .instrument import Instrument
from iop4lib.telescopes import OSNT090, OSNT150

# logging
import logging
logger = logging.getLogger(__name__)

import typing
if typing.TYPE_CHECKING:
    from iop4lib.db.reducedfit import ReducedFit, RawFit

class OSNCCDCamera(Instrument, metaclass=ABCMeta):
    r""" Abstract class for OSN CCD cameras. """

    required_masters = ['masterbias', 'masterflat']

    @classmethod
    def classify_juliandate_rawfit(cls, rawfit):
        """
        Andor cameras fits has JD keyword
        """
        import astropy.io.fits as fits
        jd = fits.getheader(rawfit.filepath, ext=0)["JD"]
        rawfit.juliandate = jd

    @classmethod
    def classify_imgtype_rawfit(cls, rawfit):
        """
        Andor cameras fits has IMAGETYP keyword: FLAT, BIAS, LIGHT

        .. note::
        **Sometimes, the IMAGETYP keyword is wrong**, it has LIGHT on it but the filename and the OBJECT keyword contain the word "Flat". In those ocassions, it should be classified as
        a FLAT.
        """
        from iop4lib.db.rawfit import RawFit
        import astropy.io.fits as fits

        with fits.open(rawfit.filepath) as hdul:
            if hdul[0].header['IMAGETYP'] == 'FLAT':
                rawfit.imgtype = IMGTYPES.FLAT
            elif hdul[0].header['IMAGETYP'] == 'BIAS':
                rawfit.imgtype = IMGTYPES.BIAS
            elif hdul[0].header['IMAGETYP'] == 'LIGHT':
                rawfit.imgtype = IMGTYPES.LIGHT
                # workarounds for wrong keyword in OSN (see note in docstring)
                if "FLAT" in hdul[0].header["OBJECT"].upper() and "FLAT" in rawfit.filename.upper():
                    rawfit.imgtype = IMGTYPES.FLAT
                elif "BIAS" in hdul[0].header["OBJECT"].upper() and "BIAS" in rawfit.filename.upper():
                    rawfit.imgtype = IMGTYPES.BIAS
            else:
                logger.error(f"Unknown image type for {rawfit.fileloc}.")
                rawfit.imgtype = IMGTYPES.ERROR
                raise ValueError

    @classmethod
    def classify_band_rawfit(cls, rawfit):
        """
            OSN Files have no FILTER keyword if they are BIAS, FILTER=Clear if they are FLAT, and FILTER=FilterName if they are LIGHT.
            For our DB, we have R, U, ..., None, ERROR.

            For polarimetry, which is done by taking four images with the R filter at different angles, we have R_45, R0, R45, R90.
        """

        from iop4lib.db.rawfit import RawFit

        if 'FILTER' not in rawfit.header:
            if rawfit.imgtype == IMGTYPES.BIAS:
                rawfit.band = BANDS.NONE
            else:
                rawfit.band = BANDS.ERROR
                raise ValueError(f"Missing FILTER keyword for {rawfit.fileloc} which is not a bias (it is a {rawfit.imgtype}).")
        elif rawfit.header['FILTER'] in {"Clear", ""}:  
            if rawfit.imgtype == IMGTYPES.FLAT:
                rawfit.band = BANDS.NONE
            else:
                rawfit.band = BANDS.ERROR
                raise ValueError(f"FILTER keyword is 'Clear' for {rawfit.fileloc} which is not a flat (it is a {rawfit.imgtype}).")
        else:
            rawfit.band = rawfit.header['FILTER'][0] # First letter of the filter name (R, U, ...) includes cases as R45, R_45, etc   

    @classmethod
    def classify_obsmode_rawfit(cls, rawfit):
        """
        In OSN Andor Polarimetry, we only have polarimetry for filter R, and it is indicated as R_45, R0, R45, R90 (-45, 0, 45 and 90 degrees). They correspond
        to the different angles of the polarimeter.

        For photometry, the filter keyword willl be simply the letter R, U, etc.

        The values for angles are -45, 0, 45 and 90.

        Lately we have seen "R-45" instead of "R_45", so we have to take care of that too.
        """

        from iop4lib.db.rawfit import RawFit
        import re

        if rawfit.band == BANDS.ERROR:
            raise ValueError("Cannot classify obsmode if band is ERROR.")
        
        if rawfit.band == BANDS.R:
            if rawfit.header['FILTER'] == "R":
                rawfit.obsmode = OBSMODES.PHOTOMETRY
            else:
                logger.debug("Band is R, but FILTER is not exactly R, for OSN this must mean it is polarimetry. Trying to extract angle from FILTER keyword.")

                rawfit.obsmode = OBSMODES.POLARIMETRY

                if rawfit.header['FILTER'] == "R_45" or rawfit.header['FILTER'] == "R-45":
                    rawfit.rotangle = -45
                elif rawfit.header['FILTER'] == "R0":
                    rawfit.rotangle = 0
                elif rawfit.header['FILTER'] == "R45" or rawfit.header['FILTER'] == "R+45":
                    rawfit.rotangle = 45
                elif rawfit.header['FILTER'] == "R90":
                    rawfit.rotangle = 90
                else:
                    raise ValueError(f"Cannot extract angle from FILTER keyword '{rawfit.header['FILTER']}'.")
        else:
            logger.debug("Band is not R, assuming it is photometry.")
            rawfit.obsmode = OBSMODES.PHOTOMETRY

    @classmethod
    def get_header_hintcoord(cls, rawfit):
        """ Get the position hint from the fits header as a SkyCoord.

        OSN T090 / AndorT090 have keywords OBJECT, OBJECTRA, OBJECTDEC in the header; e.g: 
        OBJECT 	TXS0506
        OBJCTRA 	05 09 20 ---> this can be input with unit u.hourangle
        OBJCTDEC 	+05 41 16 ---> this can be input with unit u.deg
        """     
        
        hint_coord = SkyCoord(Angle(rawfit.header['OBJCTRA'], unit=u.hourangle), Angle(rawfit.header['OBJCTDEC'], unit=u.degree), frame='icrs')
        return hint_coord
    
    @classmethod
    def get_astrometry_position_hint(cls, rawfit, allsky=False, n_field_width=1.5, hintsep=None):
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
        
        if allsky:
            hintsep = 180.0 * u.deg
        else:
            if hintsep is None:
                hintsep = (n_field_width * cls.field_width_arcmin*u.Unit("arcmin"))

        return astrometry.PositionHint(ra_deg=hintcoord.ra.deg, dec_deg=hintcoord.dec.deg, radius_deg=hintsep.to_value("deg"))
    
    @classmethod
    def get_astrometry_size_hint(cls, rawfit):
        """ Get the size hint for this telescope / rawfit.

            According to OSN T090 camera information (https://www.osn.iaa.csic.es/page/camaras-ccdt150-y-ccdt90) 
            the camera pixels are 0.387as/px and it has a field of view of 13,20' x 13,20'. So we provide close values 
            for the hint. If the files are 1x1 it will be 0.387as/px, if 2x2 it will be twice.

            According to OSN T0150 camera information (https://www.osn.iaa.csic.es/page/camaras-ccdt150-y-ccdt90) 
            camera pixels are 0.232as/px and it has a field of view of 7.92' x 7.92'.
            If the files are 1x1 it will be that, if they are 2x2 it will be twice.
        """

        if rawfit.header['NAXIS1'] == 2048:
            return astrometry.SizeHint(lower_arcsec_per_pixel=0.95*cls.arcsec_per_pix, upper_arcsec_per_pixel=1.05*cls.arcsec_per_pix)
        elif rawfit.header['NAXIS1'] == 1024:
            return astrometry.SizeHint(lower_arcsec_per_pixel=2*0.95*cls.arcsec_per_pix, upper_arcsec_per_pixel=2*1.05*cls.arcsec_per_pix)
            
    @classmethod
    def has_pairs(cls, fit_instance):
        return False

    @classmethod
    def compute_relative_polarimetry(cls, polarimetry_group):
        """ Computes the relative polarimetry for a polarimetry group for OSNT090 observations.

        .. note:: 
            The rotation angle in OSNT090 refers to the angle between the polarized filter and some reference direction. This is different
            to the rotation angle for CAHA-T220 which is the angle between the half-wave plate (HW) and its fast (extraordinary) axis. See the docs
            for ``CAHAT220.compute_relative_polarimetry`` for more information.
        
        Instrumental polarization is corrected. Currently values are hardcoded in qoff, uoff, dqoff, duoff, Phi, dPhi (see code),
        but the values without any correction are stored in the DB so the correction can be automatically obtained in the future.
        """
                
        from iop4lib.db.aperphotresult import AperPhotResult
        from iop4lib.db.photopolresult import PhotoPolResult

        logger.debug("Computing %s relative polarimetry for group: %s", cls.name, "".join(map(str,polarimetry_group)))

        # Instrumental polarization

        ## values for T090 TODO: update them  manually or (preferibly) dinamically (TODO)
        ## to compute the instrumental polarization we need to get the mean of the Q and U images, use zero
        ## (done in the _X_nocorr variables)

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
        rot_angles_required = {0.0, 45.0, 90.0, -45.0}

        if not rot_angles_required.issubset(rot_angles_available):
            logger.error(f"Rotation angles missing: {rot_angles_required - rot_angles_available}; returning early.")
            return

        # 1. Compute all aperture photometries

        aperpix, r_in, r_out, fit_res_dict = cls.estimate_common_apertures(polarimetry_group, reductionmethod=REDUCTIONMETHODS.RELPHOT)
        target_fwhm = fit_res_dict['mean_fwhm']
        
        logger.debug(f"Computing aperture photometries for the {len(polarimetry_group)} reducedfits in the group with target {aperpix:.1f}.")

        for reducedfit in polarimetry_group:
            cls.compute_aperture_photometry(reducedfit, aperpix, r_in, r_out)

        # 2. Compute relative polarimetry for each source (uses the computed aperture photometries)

        logger.debug("Computing relative polarimetry.")

        photopolresult_L = list()

        for astrosource in group_sources:
            
            flux_0 = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, aperpix=aperpix, pairs="O", reducedfit__rotangle=0.0).flux_counts
            flux_0_err = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", aperpix=aperpix, reducedfit__rotangle=0.0).flux_counts_err

            flux_45 = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", aperpix=aperpix, reducedfit__rotangle=45.0).flux_counts
            flux_45_err = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", aperpix=aperpix, reducedfit__rotangle=45.0).flux_counts_err

            flux_90 = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", aperpix=aperpix, reducedfit__rotangle=90.0).flux_counts
            flux_90_err = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", aperpix=aperpix, reducedfit__rotangle=90.0).flux_counts_err

            flux_n45 = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", aperpix=aperpix, reducedfit__rotangle=-45.0).flux_counts
            flux_n45_err = AperPhotResult.objects.get(reducedfit__in=polarimetry_group, astrosource=astrosource, pairs="O", aperpix=aperpix, reducedfit__rotangle=-45.0).flux_counts_err

            # from IOP3 polarimetry_osn() :

            fluxes = np.array([flux_0, flux_45, flux_90, flux_n45])
            flux_mean = fluxes.mean()
            flux_err = fluxes.std() / math.sqrt(len(fluxes))
            
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
 
            dqa = qc*math.cos(2*Phi) * math.sqrt((dqc/qc)**2+((2*dPhi*math.sin(2*Phi))/(math.cos(2*Phi)))**2) 
            dqb = uc*math.sin(2*Phi) * math.sqrt((duc/uc)**2+((2*dPhi*math.cos(2*Phi))/(math.sin(2*Phi)))**2)
            dua = qc*math.sin(2*Phi) * math.sqrt((dqc/qc)**2+((2*dPhi*math.cos(2*Phi))/(math.sin(2*Phi)))**2) 
            dub = uc*math.cos(2*Phi) * math.sqrt((duc/uc)**2+((2*dPhi*math.sin(2*Phi))/(math.cos(2*Phi)))**2)
            
            dq = np.sqrt(dqa**2+dqb**2)
            du = np.sqrt(dua**2+dub**2)
            
            P = math.sqrt(q**2 + u**2)
            dP = P * (1/(q**2+u**2)) * math.sqrt((q*dq)**2+(u*du)**2)

            Theta_0 = 0
            Theta = (1/2) * math.degrees(math.atan2(u,q) + Theta_0)
            dTheta = (0.5 * 180.0 / math.pi) * dP/P

            # compute also non-corrected values for computation of instrumental polarization

            _Phi_nocorr = 0 # no rotation correction?
            _qc_nocorr = qraw # no offset correction
            _uc_nocorr = uraw # no offset correction
            _q_nocorr = _qc_nocorr*math.cos(2*_Phi_nocorr) - _uc_nocorr*math.sin(2*_Phi_nocorr)
            _u_nocorr = _qc_nocorr*math.sin(2*_Phi_nocorr) + _uc_nocorr*math.cos(2*_Phi_nocorr) 
            _p_nocorr = math.sqrt(_q_nocorr**2 + _u_nocorr**2)
            _Theta_0_nocorr = 0
            _Theta_nocorr = (1/2) * math.degrees(math.atan2(_u_nocorr,_q_nocorr) + _Theta_0_nocorr)
            _x_px, _y_px = astrosource.coord.to_pixel(polarimetry_group[0].wcs)
            
            # compute instrumental magnitude (same as for CAHA)

            if flux_mean <= 0.0:
                logger.warning(f"{polarimetry_group=}: negative flux mean encountered while relative polarimetry for {astrosource=} ??!! It will be nan, but maybe we should look into this...")

            mag_inst = -2.5 * np.log10(flux_mean) # slower than math.log10 but returns nan when flux < 0 instead of throwing error (see https://github.com/juanep97/iop4/issues/24)
            mag_inst_err = math.fabs(2.5 / math.log(10) * flux_err / flux_mean)

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
                                           flux_counts=flux_mean, p=P, p_err=dP, chi=Theta, chi_err=dTheta,
                                           _x_px=_x_px, _y_px=_y_px, _q_nocorr=_q_nocorr, _u_nocorr=_u_nocorr, _p_nocorr=_p_nocorr, _chi_nocorr=_Theta_nocorr,
                                           aperpix=aperpix)
            
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


class AndorT90(OSNCCDCamera):
    
    name = "AndorT90"
    telescope = OSNT090.name

    instrument_kw_L = ["AndorT90", "RoperT90"] # RoperT90 only in some old images before the PC was updated after camera replacement

    field_width_arcmin = 13.2 
    field_height_arcmin = 13.2
    arcsec_per_pix = 0.387
    gain_e_adu = 4.5

    @classmethod
    def build_wcs(self, reducedfit: 'ReducedFit', *args, **kwargs):
        r""" Overriden for OSN-T090
         
        Overriden to account for HIP2838 U band images with exptime < 2.5, which are known to fail.
        """
        from iop4lib.utils.astrometry import BuildWCSResult

        # OSN-T090 images of HIP2838 in band U with < 2.5s
        # they can not get automatically calibrated because there are almost no sources visible, just return error so we dont loose time trying parameters.

        if reducedfit.band == "U" and "HIP2838" in reducedfit.filename and reducedfit.exptime < 2.5:
            logger.error("Skipping WCS build for HIP2838 U band image with exptime < 2.5 , as it is known to fail, and we will only lose time. Manual calibration is needed for this image. See build_wcs_for_HIP2838_U_band_images.ipynb for more info.")
            return BuildWCSResult(success=False)
        
        return super().build_wcs(reducedfit, *args, **kwargs)

class AndorT150(OSNCCDCamera):
        
    name = "AndorT150"
    telescope = OSNT150.name

    instrument_kw_L = ["Andor", "AndorT150"]

    arcsec_per_pix = 0.232
    gain_e_adu = 4.5
    field_width_arcmin = 7.92
    field_height_arcmin = 7.92


class RoperT90(OSNCCDCamera):
    name = "RoperT90"
    telescope = OSNT090.name

    instrument_kw_L = ["RoperT90"]

    arcsec_per_pix = 0.387
    gain_e_adu = 7.14
    field_width_arcmin = 13.2
    field_height_arcmin = 13.2

