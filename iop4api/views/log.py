# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.http import StreamingHttpResponse
from django.contrib.auth.decorators import permission_required
from django.contrib.admin.views.decorators import staff_member_required

# iop4lib imports

# other imports
import coloredlogs
import coloredlogs.converter

#logging
import logging
logger = logging.getLogger(__name__)


def _log_file_generator():
    with open(iop4conf.log_fname, "r") as f:
        for line in f:
            yield coloredlogs.converter.convert(line)

@staff_member_required
@permission_required(["iop4api.view_photpolresult", "iop4api.view_astrosource"]) 
def log(request):
    r"""Staff memember required. Since the log file can be very large, we use a generator to stream it to the client."""
    return StreamingHttpResponse(_log_file_generator(), content_type="text/plain")