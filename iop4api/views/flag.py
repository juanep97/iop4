# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import permission_required
from django.views.decorators.csrf import csrf_exempt

# iop4lib imports
from iop4lib.db import PhotoPolResult
import json

# other imports

#logging
import logging
logger = logging.getLogger(__name__)


@csrf_exempt
@permission_required(["iop4api.change_photopolresult"])
def flag(request):

    try:
        payload = json.loads(request.body)
        flag = payload['flag']
        pks = payload['pk_array']
        vals = payload['vals']
    except:
        return HttpResponseBadRequest("Could not parse JSON")
    
    qs = PhotoPolResult.objects.filter(pk__in=pks)

    for r in qs:
        if vals[pks.index(r.pk)]:
            r.set_flag(flag)
        else:
            r.unset_flag(flag)

    PhotoPolResult.objects.bulk_update(qs, ['flags'])

    new_flag_dict = {k:v for k,v in PhotoPolResult.objects.filter(pk__in=pks).values_list('pk', 'flags')}

    logger.debug(f"{new_flag_dict=}")

    return JsonResponse({'success': True, 'flag': flag, 'new_flag_dict':new_flag_dict})

