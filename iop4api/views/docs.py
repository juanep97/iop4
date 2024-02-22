import iop4lib
iop4conf = iop4lib.Config(config_db=False)
from django.conf import settings
from django.http import Http404, FileResponse, HttpResponseForbidden, HttpResponseNotFound
from pathlib import Path
import mimetypes
import urllib

import logging
logger = logging.getLogger(__name__)

def docs(request, file_path=None):

    if file_path is None:
        file_path = "index.html"
    
    file_path = urllib.parse.unquote(file_path) 

    logger.debug(f"RECV: {file_path}")

    file_path = Path(iop4conf.basedir) / "docs" / "_build" / "html" / file_path

    if file_path.is_dir():
        file_path = file_path / "index.html"

    if settings.DEBUG:
        if file_path.exists():
            content_type = mimetypes.guess_type(file_path)[0]
            logger.debug(f"docs: {file_path} {content_type}")
            response = FileResponse(open(file_path, 'rb'), content_type=content_type)
            return response
        else:
            logger.debug(f"docs: {file_path} not found")
            return HttpResponseNotFound("Not found")
    else:
        return HttpResponseForbidden("Forbidden")