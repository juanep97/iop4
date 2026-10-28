# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# other imports
from .instrument import Instrument

# logging
import logging
logger = logging.getLogger(__name__)


class RoboPol(Instrument):
    """Stub instrument class for the RoboPol polarimeter (Skinakas Observatory, Greece).

    This class registers RoboPol in IOP4 so that data imported from external
    RoboPol observations are colour-coded and listed in the interactive plot
    legend.  Full reduction support can be added later by implementing the
    corresponding class methods from Instrument.
    """

    name = "robopol"
    telescope = "Skinakas"

    instrument_kw_L = ["robopol"]

    arcsec_per_pix = None
    gain_e_adu = None
    field_width_arcmin = None
    field_height_arcmin = None

    required_masters = []
