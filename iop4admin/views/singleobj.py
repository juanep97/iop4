# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# iop4lib imports
from iop4admin.sites import iop4admin_site

# django imports
from django.views.generic.detail import DetailView
from django.contrib.admin.views.decorators import staff_member_required
from django.utils.decorators import method_decorator

# other imports


# logging
import logging
logger = logging.getLogger(__name__)


@method_decorator(staff_member_required, name='dispatch')
class SingleObjView(DetailView):
    template_name = 'iop4admin/view_singleobj.html'  # specify your custom template here

    def get_object(self, queryset=None):
        obj = super().get_object(queryset)
        # Add any extra object loading or checking logic here
        return obj
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context.update(iop4admin_site.each_context(self.request))

        obj = self.get_object()

        context['title'] = f"{self.model.__name__} {obj.id}"
        context['app_label'] = obj._meta.app_label
        context['clsname'] = obj.__class__.__name__
        context['model_name'] = obj._meta.model_name
        context['model_verbose_name'] = obj._meta.verbose_name
        context['model_verbose_name_plural'] = obj._meta.verbose_name_plural

        return context