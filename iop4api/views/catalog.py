# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)


# django imports
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import permission_required
from django.db import models

# iop4lib imports
from iop4lib.db import AstroSource
from iop4lib.enums import SRCTYPES
from iop4lib.utils import qs_to_table

# other imports

#logging
import logging
logger = logging.getLogger(__name__)

# API VIEWS



@permission_required(["iop4api.view_astrosource"])
def catalog(request):
    qs = AstroSource.objects.exclude(srctype=SRCTYPES.CALIBRATOR)

    all_column_names = [f.name for f in AstroSource._meta.get_fields() if hasattr(f, 'verbose_name')]

    return JsonResponse(qs_to_table(qs=qs, column_names=all_column_names))
