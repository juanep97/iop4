# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# other imports
import numpy as np

# iop4lib imports
from iop4lib.instruments.dipol import DIPOL
from iop4lib.telescopes import OSNT150

# logging
import logging
logger = logging.getLogger(__name__)

from iop4lib.typing import *

class PORTENT(DIPOL):
    name = "PORTENT"
    telescope = OSNT150.name

    instrument_kw_L = ["PORTENT"]

    reference_binning = 4 # 4x4
    arcsec_per_pix = 0.258
    field_width_arcmin = 6.71
    field_height_arcmin = 4.50 
    
    gain_e_adu = 1 #To modify

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
