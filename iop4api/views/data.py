# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import permission_required
from django.db import models

# iop4lib imports
from iop4lib.db import AstroSource, PhotoPolResult
from iop4lib.utils import qs_to_table

# other imports
from astropy.time import Time

#logging
import logging
logger = logging.getLogger(__name__)


@permission_required(["iop4api.view_photopolresult", "iop4api.view_astrosource"])
def data(request):

    source_name = request.POST.get("source_name", None)

    if not AstroSource.objects.filter(name=source_name).exists(): 
        return HttpResponseBadRequest(f"Source '{source_name}' does not exist".format(source_name=source_name))
    
    qs = PhotoPolResult.objects.filter(astrosource__name=source_name)

    data = qs.values()

    all_column_names = data[0].keys()
    default_column_names = ['juliandate', 'instrument', 'band', 'mag', 'mag_err', 'p', 'p_err', 'chi', 'chi_err']

    table_and_columns = qs_to_table(data=data, model=PhotoPolResult, column_names=all_column_names, default_column_names=default_column_names)
    
    result = {
                "data": table_and_columns["data"],
                "columns": table_and_columns["columns"],
                "query": {
                            "source_name": source_name, 
                            "count": qs.count()
                        }
    }

    # annotate with date fromt the julian date
    for r in result["data"]:
        r["date"] = Time(r["juliandate"], format='jd').iso
    result["columns"].append({"name": "date", "title": "date", "type": "date", "help": "date and time in ISO 8601 format, from the julian date"})    

    # annotate with flag labels
    for r in result["data"]:
        r["flag_labels"] = ",".join(PhotoPolResult.FLAGS.get_labels(r["flags"]))
    result["columns"].append({"name": "flag_labels", "title": "flag labels", "type": "string", "help": "flags as human readable labels"})

    return JsonResponse(result)

