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
import astropy.units as u
from photutils.aperture import CircularAperture
from astropy.wcs.utils import fit_wcs_from_points
from astropy.coordinates import SkyCoord, Angle
import itertools
import yaml
from importlib import resources
from collections.abc import Iterable
import random
from astropy.time import Time

# iop4lib imports
from iop4lib.enums import IMGTYPES, BANDS, OBSMODES, SRCTYPES
from .instrument import InstrumentHWP
from iop4lib.utils import get_angle_from_history, build_wcs_centered_on, get_simbad_sources
from iop4lib.utils.sourcedetection import get_segmentation, get_cat_sources_from_segment_map, get_bkg
from iop4lib.utils.plotting import imshow_w_sources, plot_preview_astrometry
from iop4lib.utils.astrometry import BuildWCSResult
from iop4lib.telescopes import OSNT150
from iop4lib.utils.polarization import (
    compute_stokes_HWP_fit_rel_nonideal,
    compute_stokes_HWP_fit_1pair_rel,
)

# logging
import logging
logger = logging.getLogger(__name__)

from iop4lib.typing import *


###############################################################################
class PORTENT(DIPOL):
    name = "PORTENT"
    telescope = OSNT150.name

    instrument_kw_L = ["PORTENT"]

    reference_binning = 2 # 2x2
    arcsec_per_pix = 0.134
    field_width_arcmin = 9.22
    field_height_arcmin = 6.28 
    
    gain_e_adu = 1

    required_masters = ['masterbias', 'masterflat', 'masterdark']

    # pre computed pairs distances to use in the astrometric calibrations
    # obtained from calibrated photometry fields (see CAFOS comments on how to 
    # recompute)
    
    disp_sign_mean = np.array([-209,  13]) # actually median
    disp_sign_std = np.array([12, 16]) # actually approx 0.05, 0.95 pcts
    disp_mean, disp_std = np.abs(disp_sign_mean), np.abs(disp_sign_std)

    # pre computed image angle from ~500 full photometric fields (which dont need it as input)
    # In [19]: np.quantile(angle_L,[(1-0.9973)/2,(1-0.9545)/2,(1-0.6827)/2])
    # Out[19]: array([176.85992732, 177.06814576, 177.34785346])

    # In [20]: np.quantile(angle_L,[1-(1-0.9973)/2,1-(1-0.9545)/2,1-(1-0.6827)/2])
    # Out[20]: array([181.73222701, 181.51777191, 181.36541356])

    # In [21]: np.mean(angle_L), np.median(angle_L), np.std(angle_L)
    # Out[21]: (179.08971366048235, 177.6282921156412, 1.9341471640122656)

    default_pol_method = compute_stokes_HWP_fit_rel_nonideal, compute_stokes_HWP_fit_1pair_rel

    rot_angles_expected = {0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5}
    min_rot_angles_required = 8

    default_sky_angle = 177.6
    default_sky_angle_std = 2.3

    @classmethod
    def get_instrumental_polarization(cls, reducedfit) -> InstrumentalPolarizationDict:
        """ Returns the instrumental polarization for to be used for a given reducedfit.

        The instrumental polarization is a dictionary with the following keys:
            - Q_inst: instrumental Q Stokes parameter (0-1)
            - dQ_inst: instrumental Q error (0-1)
            - U_inst: instrumental U Stokes parameter (0-1)
            - dU_inst: instrumental U error (0-1)
            - CPA: zero-angle (deg)
        """

        # compute_stokes_HWP_fit_rel_nonideal:
        instr_pol_dict = {
            'q_inst' :  +0.04 / 100,
            'dq_inst':   0.07 / 100,
            'u_inst' :  -4.55 / 100,
            'du_inst':   0.07 / 100,
            'CPA'    :  41.0,
            'dCPA'   :  3.0,
        }

        return instr_pol_dict
