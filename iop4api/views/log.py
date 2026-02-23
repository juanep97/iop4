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
from pathlib import Path

#logging
import logging
logger = logging.getLogger(__name__)


def get_log_file_generator(fpath):
    def _log_file_generator():
        with open(fpath, "r") as f:
            for line in f:
                yield coloredlogs.converter.convert(line)
    return _log_file_generator()

@staff_member_required
@permission_required(["iop4api.view_photopolresult", "iop4api.view_astrosource"])
def log(request):
    r"""Staff member required. Since the log file can be very large, we use a generator to stream it to the client."""

    if (log_fname := request.GET.get("log_file", None)):
        fpath = Path(iop4conf.datadir) / "logs" / log_fname
    else:
        fpath = Path(iop4conf.log_file)

    fpath = fpath.resolve()

    if not fpath.is_relative_to(iop4conf.datadir / "logs"):
        return HttpResponse("Access denied.", status=403)

    if not fpath.exists():
        return HttpResponse("Log file not found", status=404)
    
    return StreamingHttpResponse(get_log_file_generator(fpath), content_type="text/plain")
