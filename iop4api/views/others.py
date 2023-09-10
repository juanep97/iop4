# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)


# django imports
from django.shortcuts import render
from django.urls import reverse
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import permission_required

# iop4lib imports
from ..models import *

# other imports
import json

#logging
import logging
logger = logging.getLogger(__name__)



def index(request, tabs=None):

    context = {}

    context["tab_tree"] =  json.dumps({"about":{}, "login":{}, "explore": {"catalog":{}, "query":{}, "plot":{}, "data":{}, "log":{}}})
    
    # pass the tabs to the template (e.g. /iop4/tab1/tab2/, see urls.py)
    if tabs is not None:
        context['tabs'] = {f"C{i+1}selectedTab":tab for i, tab in enumerate(tabs)}

        # redirect to login if they are trying to see a tab that requires login
        if not request.user.is_authenticated and "C1selectedTab" in context['tabs'] and context['tabs']["C1selectedTab"] not in ["about", "login"]:
            return redirect("{}".format(reverse('iop4api:index', args=[["login",]])))

    # if the user is logged, pass source names to the template
    if request.user.is_authenticated:
        context['source_name_list'] = AstroSource.objects.exclude(srctype=SRCTYPES.CALIBRATOR).exclude(srctype=SRCTYPES.UNPOLARIZED_FIELD_STAR).values_list('name', flat="True")

    return render(request, 'iop4api/index.html', context)


# API VIEWS

def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect(reverse('iop4api:index', args=[['explore', 'catalog']]))
    
    return redirect("{}?login_failed=1".format(reverse('iop4api:index', args=[["login",]])))




def logout_view(request):
    logout(request)
    return redirect('iop4api:index') 



@permission_required(["iop4api.view_astrosource"])
def catalog(request):
    qs = AstroSource.objects.exclude(srctype=SRCTYPES.CALIBRATOR).exclude(srctype=SRCTYPES.UNPOLARIZED_FIELD_STAR).values()
    return JsonResponse({'data': list(qs)})



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

    columns = [{"title": k, "field":k, "visible": (k in default_column_names)} for k in all_column_names]

    return JsonResponse({'data': list(vals), 'tabulatorjs_columns': columns})









