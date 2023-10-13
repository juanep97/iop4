# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.http import JsonResponse, HttpResponseBadRequest

# iop4lib imports

# other imports
import subprocess as sp

#logging
import logging
logger = logging.getLogger(__name__)

def test(request):
    # run tests in the docker here, return the result as below

    cmd_build_docker = [
        "docker",
        "build",
        "-t",
        "pytest",
        ".",
    ]

    cmd_run_docker = [
        "docker",
        "run", 
        "-v",
        "$HOME/.astrometry_cache:/home/testuser/.astrometry_cache",
        "pytest",
        "-vxs"
    ]
    # FIXME: properly define were is astrometry_cache dir in server

    # Build the container
    result_build_docker = sp.run(cmd_build_docker, check=True)
    
    if result_build_docker.returncode == 0:

        # Run the container
        result_run_docker = sp.run(cmd_run_docker, check=True)

        if result_run_docker.returncode == 0:
            return JsonResponse({'data': 'test'})

        else:
            raise RuntimeError("Running docker container failed")

    else:
        raise RuntimeError("Building docker container failed")
