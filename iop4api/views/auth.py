# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)

# django imports
from django.urls import reverse
from django.shortcuts import redirect
from django.contrib.auth import authenticate, login, logout

# iop4lib imports

# other imports

#logging
import logging
logger = logging.getLogger(__name__)



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

