# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)


# django imports
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import permission_required

# iop4lib imports
from iop4lib.db import AstroSource
from iop4lib.enums import SRCTYPES

# other imports

#logging
import logging
logger = logging.getLogger(__name__)

# API VIEWS



@permission_required(["iop4api.view_astrosource"])
def catalog(request):
    qs = AstroSource.objects.exclude(srctype=SRCTYPES.CALIBRATOR).exclude(srctype=SRCTYPES.UNPOLARIZED_FIELD_STAR).values()

    data = list(qs)

    if len(data) > 0:
        all_column_names = data[0].keys()
        default_column_names = set(all_column_names)
    else:
        all_column_names = []
        default_column_names = []

    columns = [{"name": k, "title": AstroSource._meta.get_field(k).verbose_name, "field":k, "visible": (k in default_column_names)} for k in all_column_names]

    return JsonResponse({'data': data, 'columns': columns})
