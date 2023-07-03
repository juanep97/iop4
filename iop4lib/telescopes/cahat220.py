# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
from abc import ABCMeta, abstractmethod
import re
import ftplib
import logging
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
import astrometry
import numpy as np
import math

# iop4lib imports
from iop4lib.enums import *
from .telescope import Telescope

# logging
import logging
logger = logging.getLogger(__name__)

class CAHAT220(Telescope, metaclass=ABCMeta):
    """
    CAHA T220 telescope.

    CAHA has a per-program ftp login (PI user and password in config).
    The file structure when you login is a list of files as:
    {yymmdd}_CAFOS/
    where CAFOS refers to the polarimeter
    and inside each folder there are the files for that day.
    """

    # telescope identification

    name = "CAHA-T220"
    abbrv = "T220"
    telescop_kw = "CA-2.2"

    # telescope specific properties

    arcsec_per_pix = 0.530
    gain_e_adu = 1.45

    # telescope specific methods

    @classmethod
    def list_remote_epochnames(cls):
        try:
            logger.debug("Loging to CAHA FTP server.")

            ftp =  ftplib.FTP(iop4conf.caha_address)
            ftp.login(iop4conf.caha_user, iop4conf.caha_password)

            remote_dirnameL_all = ftp.nlst()
            ftp.quit()

            if '.' in remote_dirnameL_all:
                remote_dirnameL_all.remove('.')
            if '..' in remote_dirnameL_all:
                remote_dirnameL_all.remove('..')

            remote_epochnameL_all = list()

            for dirname in remote_dirnameL_all:
                matchs = re.findall(r"([0-9]{2}[0-9]{2}[0-9]{2})_CAFOS", dirname)
                if len(matchs) != 1:
                    logger.warning(f"Could not parse {dirname} as a valid epochname.")
                else:
                    remote_epochnameL_all.append(f"{cls.name}/20{matchs[0][0:2]}-{matchs[0][2:4]}-{matchs[0][4:6]}")

            logger.debug(f"Total of {len(remote_epochnameL_all)} epochs in CAHA.")

            return remote_epochnameL_all
        
        except Exception as e:
            raise Exception(f"Error listing remote epochs for {Telescope.name}: {e}.")

    @classmethod
    def list_remote_raw_fnames(cls, epoch):
        try:

            logger.debug("Loging to CAHA FTP server.")

            ftp =  ftplib.FTP(iop4conf.caha_address)
            ftp.login(iop4conf.caha_user, iop4conf.caha_password)

            try:
                logger.debug(f"Trying to change {epoch.yymmdd}_CAFOS/ directory.")
                ftp.cwd(f"{epoch.yymmdd}_CAFOS")
            except Exception as e:
                logger.debug(f"Could not change to {epoch.yymmdd}_CAFOS/ trying cd to {epoch.yymmdd}/ instead.")
                ftp.cwd(f"{epoch.yymmdd}")

            remote_fnameL_all = ftp.nlst()
            ftp.quit()

            if '.' in remote_fnameL_all:
                remote_fnameL_all.remove('.')
            if '..' in remote_fnameL_all:
                remote_fnameL_all.remove('..')

            logger.debug(f"Total of {len(remote_fnameL_all)} files in CAHA {epoch.epochname}.")

            remote_fnameL = [s for s in remote_fnameL_all if re.compile(r".*\.fits?").search(s)] # Filter by filename pattern (get only our files)
            
            logger.debug(f"Filtered to {len(remote_fnameL)} *.fit(s) files in CAHA {epoch.epochname}.")

            # all files are ours
            return remote_fnameL
        
        except Exception as e:
            raise Exception(f"Error listing remote dir for {epoch.epochname}: {e}.")
        
    @classmethod
    def download_rawfits(cls, rawfits):
        try:
            ftp =  ftplib.FTP(iop4conf.caha_address)
            ftp.login(iop4conf.caha_user, iop4conf.caha_password)

            for rawfit in rawfits:
                #logger.debug(f"Changing to CAHA dir {rawfit.epoch.yymmdd}_CAFOS ...")

                ftp.cwd(f"{rawfit.epoch.yymmdd}_CAFOS")

                logger.debug(f"Downloading {rawfit.fileloc} from CAHA archive ...")

                with open(rawfit.filepath, 'wb') as f:
                    ftp.retrbinary('RETR ' + rawfit.filename, f.write)

                ftp.cwd("..")

            ftp.quit()
        except Exception as e:
            raise Exception(f"Error downloading {rawfits}: {e}.")

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
            else:
                logger.error(f"Unknown image type for {rawfit.fileloc}.")
                rawfit.imgtype = IMGTYPES.ERROR
                raise ValueError
            
    @classmethod
    def classify_band_rawfit(cls, rawfit):
        """
        INSFLNAM is BesselR ??
        """

        from iop4lib.db.rawfit import RawFit
        import astropy.io.fits as fits

        if 'INSFLNAM' in rawfit.header:
            if rawfit.header['INSFLNAM'] == 'BessellR':
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
                logger.debug(f"Probabbly not important, but {rawfit.fileloc} is BIAS but has polarimetry keywords, does it makes sense?")
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
    def get_astrometry_position_hint(cls, rawfit, allsky=False, n_field_width=1.5):
        """ Get the position hint from the FITS header as an astrometry.PositionHint object. """        

        hintcoord = cls.get_header_hintcoord(rawfit)

        if allsky:
            hintsep = 180
        else:
            hintsep = n_field_width * u.Quantity("16 arcmin").to_value(u.deg) # 16 arcmin is the full field size of the CAFOS T2.2, our cut is smaller (6.25, 800x800, but the pointing kws might be from anywhere in the full field)

        return astrometry.PositionHint(ra_deg=hintcoord.ra.deg, dec_deg=hintcoord.dec.deg, radius_deg=hintsep)
    
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
    def compute_relative_polarimetry(cls, polarimetry_group):
        """ Computes the relative polarimetry for a polarimetry group for CAHA T220 observations."""
        
        from iop4lib.db.aperphotresult import AperPhotResult
        from iop4lib.db.photopolresult import PhotoPolResult

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

            aperpix = astrosource.get_aperpix()

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

            # if np.any(fluxes_O <= 0) or np.any(fluxes_E <= 0):
            #     logger.warning(f"{astrosource}: fluxes <= 0 !!")
            #     logger.debug(f"Fluxes_O: {fluxes_O}")
            #     logger.debug(f"Fluxes_E: {fluxes_E}")

            fluxes = (fluxes_O + fluxes_E) /2.
            flux_mean = fluxes.mean()
            flux_std = fluxes.std()

            RQ = np.sqrt((flux_O_0 / flux_E_0) / (flux_O_45 / flux_E_45))
            dRQ = RQ * np.sqrt((flux_O_0_err / flux_O_0) ** 2 + (flux_E_0_err / flux_E_0) ** 2 + (flux_O_45_err / flux_O_45) ** 2 + (flux_E_45_err / flux_E_45) ** 2)

            RU = np.sqrt((flux_O_0 / flux_E_22) / (flux_O_67 / flux_E_67))
            dRU = RU * np.sqrt((flux_O_22_err / flux_O_22) ** 2 + (flux_E_22_err / flux_E_22) ** 2 + (flux_O_67_err / flux_O_67) ** 2 + (flux_E_67_err / flux_E_67) ** 2)
        
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

