# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.http import JsonResponse
from django.contrib.auth.decorators import permission_required

# iop4lib imports
from iop4lib.db import AstroSource

# logging
import logging
logger = logging.getLogger(__name__)


@permission_required(["iop4api.view_astrosource", "iop4api.view_photopolresult"], raise_exception=True)
def last_mag(request):
    """Return the last R-band magnitude for all non-calibrator sources."""

    sources = AstroSource.objects.exclude(is_calibrator=True)

    result = []
    for src in sources:
        try:
            mag_r_avg, mag_r_avg_err, last_night = src.last_night_mag_R
        except (ZeroDivisionError, Exception) as e:
            logger.debug(f"Could not compute last_night_mag_R for {src.name}: {e}")
            mag_r_avg, mag_r_avg_err, last_night = None, None, None

        result.append({
            "name": src.name,
            "srctype": src.srctype,
            "last_mag_R": float(mag_r_avg) if mag_r_avg is not None else None,
            "last_mag_R_err": float(mag_r_avg_err) if mag_r_avg_err is not None else None,
            "last_night": str(last_night) if last_night is not None else None,
        })

    return JsonResponse({"data": result})
