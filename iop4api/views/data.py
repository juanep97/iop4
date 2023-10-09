# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import permission_required

# iop4lib imports
from iop4lib.db import AstroSource, PhotoPolResult

# other imports

#logging
import logging
logger = logging.getLogger(__name__)


@permission_required(["iop4api.view_photopolresult", "iop4api.view_astrosource"])
def data(request):

    source_name = request.POST.get("source_name", None)
    band = request.POST.get("band", "R")

    if not AstroSource.objects.filter(name=source_name).exists(): 
        return HttpResponseBadRequest(f"Source '{source_name}' does not exist".format(source_name=source_name))
    
    vals = PhotoPolResult.objects.filter(astrosource__name=source_name, band=band).values()

    if len(vals) > 0:
        all_column_names = vals[0].keys()
        default_column_names = set(all_column_names).intersection(['juliandate', 'instrument', 'band', 'mag', 'mag_err', 'p', 'p_err', 'chi', 'chi_err'])
    else:
        all_column_names = []
        default_column_names = []

    columns = [{"name": k, "title": PhotoPolResult._meta.get_field(k).verbose_name, "field":k, "visible": (k in default_column_names)} for k in all_column_names]

    return JsonResponse({'data': list(vals), 'columns': columns})

