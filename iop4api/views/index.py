# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.shortcuts import render
from django.urls import reverse
from django.shortcuts import render, redirect

# iop4lib imports
from iop4lib.db import AstroSource
from iop4lib.enums import SRCTYPES

# other imports
import json
import os
import subprocess
from pathlib import Path

#logging
import logging
logger = logging.getLogger(__name__)

GIT_COMMIT_HASH = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.realpath(__file__))).decode('ascii').strip()
GIT_BRANCH = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=os.path.dirname(os.path.realpath(__file__))).decode('ascii').strip()
GIT_DESCRIBE = subprocess.check_output(['git', 'describe', '--always'], cwd=os.path.dirname(os.path.realpath(__file__))).decode('ascii').strip()

def index(request, tabs=None):

    context = {}

    tab_tree = {"about":{}, "login":{}, "explore": {"catalog":{}, "query":{}, "plot":{}, "data":{}, "logs":{}}}
    context["tab_tree"] =  json.dumps(tab_tree)
    
    # pass the tabs to the template (e.g. /iop4/tab1/tab2/, see urls.py)
    if tabs is not None:
        context['tabs'] = {f"C{i+1}selectedTab":tab for i, tab in enumerate(tabs)}

        # redirect to login if they are trying to see a tab that requires login
        if not request.user.is_authenticated and "C1selectedTab" in context['tabs'] and context['tabs']["C1selectedTab"] not in ["about", "login"]:
            return redirect("{}".format(reverse('iop4api:index', args=[["login",]])))
        
        # # # if the tab is not in the tab tree, redirect to the index
        # for i, tab in enumerate(tabs):
        #     if i == 0 and tab not in tab_tree.keys():
        #         return redirect("{}".format(reverse('iop4api:index')))

    # if the user is logged, pass source names to the template
    if request.user.is_authenticated:
        context['source_name_list'] = AstroSource.objects.exclude(srctype=SRCTYPES.CALIBRATOR).values_list('name', flat="True")

    # add the hash of the current commit installed 
    context['git_commit_hash'] = GIT_COMMIT_HASH
    context['git_branch'] = GIT_BRANCH
    context['git_describe'] = GIT_DESCRIBE

    # add the available log files
    context['log_files'] = json.dumps(sorted([os.path.basename(f) for f in os.listdir(Path(iop4conf.datadir) / "logs") if f.endswith(".log")]))

    return render(request, 'iop4api/index.html', context)
