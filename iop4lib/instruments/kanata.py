# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# other imports
from .instrument import Instrument

# logging
import logging
logger = logging.getLogger(__name__)


class Kanata(Instrument):
    """Stub instrument class for the Kanata telescope (Hiroshima University, Japan).

    This class registers Kanata in IOP4 so that data imported from external
    Kanata observations are colour-coded and listed in the interactive plot
    legend.  Full reduction support can be added later by implementing the
    corresponding class methods from Instrument.
    """

    name = "kanata"
    telescope = "kanata"

    instrument_kw_L = ["kanata"]

    arcsec_per_pix = None
    gain_e_adu = None
    field_width_arcmin = None
    field_height_arcmin = None

    required_masters = []

    filters = {
        "R" : "bessell",
        "V" : ""
    }