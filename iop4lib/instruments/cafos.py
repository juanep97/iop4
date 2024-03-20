# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
import astrometry
import numpy as np
import math

# iop4lib imports
from iop4lib.enums import *
from .instrument import Instrument
from iop4lib.telescopes import CAHAT220

# logging
import logging
logger = logging.getLogger(__name__)

import typing
if typing.TYPE_CHECKING:
    from iop4lib.db import RawFit, ReducedFit, Epoch

class CAFOS(Instrument):
        
    name = "CAFOS2.2"
    telescope = CAHAT220.name

    instrument_kw_L = ["CAFOS 2.2"]

    arcsec_per_pix = 0.530
    gain_e_adu = 1.45
    field_width_arcmin = 34.0
    field_height_arcmin = 34.0

    required_masters = ['masterbias', 'masterflat']

    # pre computed pairs distances to use in the astrometric calibrations
    # obtained from calibrated fields
    
    # computed with:
    # > In [1]: qs = ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.BUILT_REDUCED, instrument="CAFOS2.2").all()
    # > In [2]: disp_sign_mean = np.mean([redf.astrometry_info[-1]['seg_disp_sign'] for redf in qs[len(qs)-300:len(qs)-1]], axis=0)
    # > In [3]: disp_sign_std = np.std([redf.astrometry_info[-1]['seg_disp_sign'] for redf in qs[len(qs)-300:len(qs)-1]], axis=0)

    disp_sign_mean, disp_sign_std = np.array([-35.72492116, -0.19719535]), np.array([1.34389, 1.01621491])
    disp_mean, disp_std = np.abs(disp_sign_mean), disp_sign_std


    @classmethod
    def classify_juliandate_rawfit(cls, rawfit):
        """
        CAHA T220 has a DATE keyword in the header in ISO format.
        """
        import astropy.io.fits as fits
        from astropy.time import Time 
        date = fits.getheader(rawfit.filepath, ext=0)["DATE"]
        jd = Time(date, format='isot', scale='utc').jd
        rawfit.juliandate = jd

    @classmethod
    def classify_imgtype_rawfit(cls, rawfit):
        """
        CAHA T220 has a IMAGETYP keyword in the header: flat, bias, science
        """
        from iop4lib.db.rawfit import RawFit
        import astropy.io.fits as fits

        with fits.open(rawfit.filepath) as hdul:
            if hdul[0].header['IMAGETYP'] == 'flat':
                rawfit.imgtype = IMGTYPES.FLAT
            elif hdul[0].header['IMAGETYP'] == 'bias':
                rawfit.imgtype = IMGTYPES.BIAS
            elif hdul[0].header['IMAGETYP'] == 'science':
                rawfit.imgtype = IMGTYPES.LIGHT
            elif hdul[0].header['IMAGETYP'] == 'dark':
                rawfit.imgtype = IMGTYPES.DARK
            else:
                logger.error(f"Unknown image type for {rawfit.fileloc}.")
                rawfit.imgtype = IMGTYPES.ERROR
                raise ValueError
            
    @classmethod
    def classify_band_rawfit(cls, rawfit):
        """
        Older data (e.g. 2007): INSFLNAM = 'John R' or INSFLNAM = 'Cous R'
        New data (e.g. 2022): INSFLNAM = 'BessellR'

        There are also images in the archive with INSFLNAM = 'John V' and INSFLNAM = 'John I', and INSFLNAM = 'free'
        """

        from iop4lib.db.rawfit import RawFit
        import astropy.io.fits as fits

        if 'INSFLNAM' in rawfit.header:
            if (rawfit.header['INSFLNAM'] == 'BessellR' or 
                rawfit.header['INSFLNAM'] == 'John R' or 
                rawfit.header['INSFLNAM'] == 'Cous R' or rawfit.header['INSFLNAM'] == 'CousinsR'):
                rawfit.band = BANDS.R
            else:
                logger.error(f"{rawfit}: unknown filter {rawfit.header['INSFLNAM']}.")
                rawfit.band = BANDS.ERROR
                raise ValueError(f"{rawfit}: unknown filter {rawfit.header['INSFLNAM']}.")
        else: 
            rawfit.band = BANDS.ERROR
            raise ValueError(f"{rawfit}: INSFLNAM keyword not present.")

    @classmethod
    def classify_obsmode_rawfit(cls, rawfit):
        """
        For CAHA T220, if we are dealing with polarimetry, we have:
        INSTRMOD:	Polarizer
        INSPOFPI 	Wollaston
        INSPOROT 	0.0, 22.48, 67.48
        
        I HAVE NOT FOUND YET OTHER VALUES THAT ARE NOT THIS, PRINT A WARNING OTHERWISE.
        """
        from iop4lib.db.rawfit import RawFit

        if rawfit.header['INSTRMOD'] == 'Polarizer' and rawfit.header['INSPOFPI'] == 'Wollaston':
            rawfit.obsmode = OBSMODES.POLARIMETRY
            rawfit.rotangle = float(rawfit.header['INSPOROT'])

            if rawfit.imgtype == IMGTYPES.BIAS:
                logger.debug(f"Probably not important, but {rawfit.fileloc} is BIAS but has polarimetry keywords, does it makes sense?")
        else:
            logger.error("Not implemented, please check the code.")

    @classmethod
    def get_header_hintcoord(cls, rawfit):
        """ Get the position hint from the FITS header as a coordinate.

        Images from CAFOS T2.2 have RA, DEC in the header, both in degrees.
        """

        hint_coord = SkyCoord(Angle(rawfit.header['RA'], unit=u.deg), Angle(rawfit.header['DEC'], unit=u.deg), frame='icrs')
        return hint_coord
    
    @classmethod
    def get_astrometry_position_hint(cls, rawfit, allsky=False, n_field_width=1.5, hintsep=None):
        """ Get the position hint from the FITS header as an astrometry.PositionHint object. 

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
                hintsep = n_field_width * u.Quantity("16 arcmin") # 16 arcmin is the full field size of the CAFOS T2.2, our cut is smaller (6.25, 800x800, but the pointing kws might be from anywhere in the full field)

        return astrometry.PositionHint(ra_deg=hintcoord.ra.deg, dec_deg=hintcoord.dec.deg, radius_deg=hintsep.to_value(u.deg))
    
    @classmethod
    def get_astrometry_size_hint(cls, rawfit):
        """ Get the size hint for this telescope / rawfit.

        from http://w3.caha.es/CAHA/Instruments/CAFOS/cafos22.html
        pixel size in arcmin is around : ~0.530 arcsec
        field size (diameter is) 16.0 arcmin (for 2048 pixels)
        it seems that this is for 2048x2048 images, our images are 800x800 but the fitsheader DATASEC
        indicates it is a cut
        """
        
        return astrometry.SizeHint(lower_arcsec_per_pixel=0.95*cls.arcsec_per_pix,  upper_arcsec_per_pixel=1.05*cls.arcsec_per_pix)

    @classmethod
    def has_pairs(cls, fit_instance):
        """ At the moment, CAFOS polarimetry. """
        return (fit_instance.obsmode == OBSMODES.POLARIMETRY)

    @classmethod
    def compute_relative_polarimetry(cls, polarimetry_group):
        """ Computes the relative polarimetry for a polarimetry group for CAFOS observations.
        
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
        rot_angles_required = {0.0, 22.48, 44.98, 67.48}

        if not rot_angles_available.issubset(rot_angles_required):
            logger.warning(f"Rotation angles missing: {rot_angles_required - rot_angles_available}")

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

            qs = AperPhotResult.objects.filter(reducedfit__in=polarimetry_group, astrosource=astrosource, aperpix=aperpix, flux_counts__isnull=False)

            equivs = ((('O',0.0),   ('E',44.98)),
                      (('O',22.48), ('E',67.48)),
                      (('O',44.98), ('E',0.0)),
                      (('O',67.48), ('E',22.48)),
                      (('E',0.0),   ('O',44.98)),
                      (('E',22.48), ('O',67.48)),
                      (('E',44.98),  ('O',0.0)),
                      (('E',67.48), ('O',22.48)))

            flux_D = dict()
            for equiv in equivs:
                if qs.filter(pairs=equiv[0][0], reducedfit__rotangle=equiv[0][1]).exists():
                    flux_D[(equiv[0][0], equiv[0][1])] = qs.filter(pairs=equiv[0][0], reducedfit__rotangle=equiv[0][1]).values_list("flux_counts", "flux_counts_err").last()
                elif qs.filter(pairs=equiv[1][0], reducedfit__rotangle=equiv[1][1]).exists():
                    logger.warning(f"Missing flux for {astrosource} {equiv[0][0]} {equiv[0][1]}, using {equiv[1][0]} {equiv[1][1]}")
                    flux_D[(equiv[0][0], equiv[0][1])] = qs.filter(pairs=equiv[1][0], reducedfit__rotangle=equiv[1][1]).values_list("flux_counts", "flux_counts_err").last()
                else:
                    logger.error(f"Missing flux for {astrosource} {equiv[0][0]} {equiv[0][1]} and {equiv[1][0]} {equiv[1][1]}")
                    return

            flux_O_0, flux_O_0_err = flux_D[('O',0.0)]
            flux_O_22, flux_O_22_err = flux_D[('O',22.48)]
            flux_O_45, flux_O_45_err = flux_D[('O',44.98)]
            flux_O_67, flux_O_67_err = flux_D[('O',67.48)]
            flux_E_0, flux_E_0_err = flux_D[('E',0.0)]
            flux_E_22, flux_E_22_err = flux_D[('E',22.48)]
            flux_E_45, flux_E_45_err = flux_D[('E',44.98)]
            flux_E_67, flux_E_67_err = flux_D[('E',67.48)]

            fluxes_O = np.array([flux_O_0, flux_O_22, flux_O_45, flux_O_67])
            fluxes_E = np.array([flux_E_0, flux_E_22, flux_E_45, flux_E_67])

            # logger.debug(f"Fluxes_O: {fluxes_O}")
            # logger.debug(f"Fluxes_E: {fluxes_E}")

            fluxes = (fluxes_O + fluxes_E) / 2.
            flux_mean = fluxes.mean()
            flux_err = fluxes.std() / math.sqrt(len(fluxes))

            RQ = np.sqrt((flux_O_0 / flux_E_0) / (flux_O_45 / flux_E_45))
            dRQ = RQ / 2 * math.sqrt((flux_O_0_err / flux_O_0) ** 2 + (flux_E_0_err / flux_E_0) ** 2 + (flux_O_45_err / flux_O_45) ** 2 + (flux_E_45_err / flux_E_45) ** 2)

            RU = np.sqrt((flux_O_22 / flux_E_22) / (flux_O_67 / flux_E_67))
            dRU = RU / 2 * math.sqrt((flux_O_22_err / flux_O_22) ** 2 + (flux_E_22_err / flux_E_22) ** 2 + (flux_O_67_err / flux_O_67) ** 2 + (flux_E_67_err / flux_E_67) ** 2)

            Q_I = (RQ - 1) / (RQ + 1)
            dQ_I = math.fabs( RQ / (RQ + 1) ** 2 * dRQ)
            U_I = (RU - 1) / (RU + 1)
            dU_I = math.fabs( RU / (RU + 1) ** 2 * dRU)

            P = math.sqrt(Q_I ** 2 + U_I ** 2)
            dP = 1/P * math.sqrt(Q_I**2 * dQ_I**2 + U_I**2 * dU_I**2)

            Theta_0 = 0
        
            if Q_I >= 0:
                Theta_0 = math.pi 
                if U_I > 0:
                    Theta_0 = -1 * math.pi
                # if Q_I < 0:
                #     Theta_0 = math.pi / 2
                
            Theta = 0.5 * math.degrees(math.atan(U_I / Q_I) + Theta_0)
            dTheta = 0.5 * 180.0 / math.pi * (1 / (1 + (U_I/Q_I) ** 2)) * math.sqrt( (dU_I/Q_I)**2 + (U_I*dQ_I/Q_I**2)**2 )

            # compute instrumental magnitude

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
                    # mag_zp_err = math.sqrt(mag_known_err ** 2 + mag_inst_err ** 2)
                    mag_zp_err = math.fabs(mag_inst_err) # do not add error on literature magnitude
            else:
                mag_zp = None
                mag_zp_err = None

            # save the results
                    
            result = PhotoPolResult.create(reducedfits=polarimetry_group, 
                                                           astrosource=astrosource, 
                                                           reduction=REDUCTIONMETHODS.RELPOL, 
                                                           mag_inst=mag_inst, mag_inst_err=mag_inst_err, mag_zp=mag_zp, mag_zp_err=mag_zp_err,
                                                           flux_counts=flux_mean, p=P, p_err=dP, chi=Theta, chi_err=dTheta,
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



    @classmethod
    def _build_shotgun_params(cls, redf: 'ReducedFit'):
        shotgun_params_kwargs = dict()

        shotgun_params_kwargs["d_eps"] = [1.0] #[1*np.linalg.norm(cls.disp_std)]
        shotgun_params_kwargs["dx_eps"] = [1.0] #[1*cls.disp_std[0]]
        shotgun_params_kwargs["dy_eps"] = [1.0] #[1*cls.disp_std[1]]
        shotgun_params_kwargs["dx_min"] = [(cls.disp_mean[0] - 5*cls.disp_std[0])]
        shotgun_params_kwargs["dx_max"] = [(cls.disp_mean[0] + 5*cls.disp_std[0])]
        shotgun_params_kwargs["dy_min"] = [(cls.disp_mean[1] - 5*cls.disp_std[1])]
        shotgun_params_kwargs["dy_max"] = [(cls.disp_mean[1] + 5*cls.disp_std[1])]
        shotgun_params_kwargs["d_min"] = [np.linalg.norm(cls.disp_mean) - 3*np.linalg.norm(cls.disp_std)]
        shotgun_params_kwargs["d_max"] = [np.linalg.norm(cls.disp_mean) + 3*np.linalg.norm(cls.disp_std)]
        shotgun_params_kwargs["bins"] = [400]
        shotgun_params_kwargs["hist_range"] = [(0,400)]

        if redf.header_hintobject is not None and redf.header_hintobject.name == "1101+384":
            shotgun_params_kwargs["bkg_filter_size"] = [3]
            shotgun_params_kwargs["bkg_box_size"] = [16]
            shotgun_params_kwargs["seg_fwhm"] = [1.0]
            shotgun_params_kwargs["npixels"] = [8, 16]
            shotgun_params_kwargs["n_rms_seg"] = [3.0, 1.5, 1.2, 1.1, 1.0]


        return shotgun_params_kwargs

    @classmethod
    def build_wcs(cls, reducedfit: 'ReducedFit', summary_kwargs : dict = None):
        return super().build_wcs(reducedfit, shotgun_params_kwargs=cls._build_shotgun_params(reducedfit), summary_kwargs=summary_kwargs)

    @classmethod
    def estimate_common_apertures(cls, reducedfits, reductionmethod=None, fit_boxsize=None, search_boxsize=(30,30), fwhm_min=2, fwhm_max=20):
        return super().estimate_common_apertures(reducedfits, reductionmethod=reductionmethod, fit_boxsize=fit_boxsize, search_boxsize=search_boxsize, fwhm_min=fwhm_min, fwhm_max=fwhm_max)