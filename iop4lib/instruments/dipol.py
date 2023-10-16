# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports

# other imports
import astrometry

# iop4lib imports
from iop4lib.enums import *
from .instrument import Instrument

# logging
import logging
logger = logging.getLogger(__name__)


class DIPOL(Instrument):

    name = "DIPOL"
    instrument_kw = "ASI Camera (1)"
    
    arcsec_per_pix = 0.134

    @classmethod
    def get_astrometry_size_hint(cls, rawfit):
        """ Get the size hint for this telescope / rawfit.

            For DIPOL in OSN-T090, according to preliminary investigation of OSN crew is:
                Las posiciones que he tomado y el ángulo de rotación en cada caso son estos:
                Dec= -10º HA=+3h  rotación=-177.3º
                Zenit rotación=-177.3º
                Dec=+60º HA=-6h rotación=-177.7º
                Dec=+70º HA=+5h rotación=-177.2º

                El campo es de 9.22 x 6.28 arcmin y el tamaño de pixel de 0.134"/pix

                El ángulo de la imagen cambia muy poco entre las posiciones muy separadas del telescopio, y es de 177.5º ± 0.3º
                Así que como mucho se produce un error de ± 0.3º en las imágenes, y el punto cero es de 2.5º.
        """

        return astrometry.SizeHint(lower_arcsec_per_pixel=0.95*cls.arcsec_per_pix, upper_arcsec_per_pixel=1.05*cls.arcsec_per_pix)