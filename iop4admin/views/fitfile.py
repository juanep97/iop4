# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# iop4lib imports
from iop4lib.db import *
from iop4lib.enums import *
from iop4api.filters import *
from .singleobj import SingleObjView
from iop4admin.sites import iop4admin_site

# django imports
from django.urls import reverse

# other imports
import os
import base64
import glob
import numpy as np
from astropy.io import fits

# logging
import logging
logger = logging.getLogger(__name__)



def get_fit_view(viewcls, modelcls):
    class _(viewcls):
        model = modelcls
    return _


class FitJS9View(SingleObjView):

    template_name = "iop4admin/view_fitviewer.html"
    
    def get_context_data(self, **kwargs):

        context = super().get_context_data(**kwargs)

        fit = self.get_object()

        context['fit'] = fit
        context['filesize_mb'] = f"{os.path.getsize(fit.filepath) / 1024 / 1024}"

        # urls

        context['fits_url'] = reverse(f'iop4admin:iop4api_{self.model._meta.model_name}_getfile', args=[fit.id])
        context['url_changelist'] = reverse('iop4admin:%s_%s_changelist' % (self.model._meta.app_label, self.model._meta.model_name)) + f"?id={fit.id}"

        if isinstance(fit, ReducedFit):
            context['url_raw'] = reverse('iop4admin:%s_%s_changelist' % (self.model._meta.app_label, fit.rawfit._meta.model_name)) + f"?id={fit.rawfit.id}"
        if isinstance(fit, RawFit) and hasattr(fit, "reduced"):
            context['url_reduced'] = reverse('iop4admin:%s_%s_changelist' % (self.model._meta.app_label, fit.reduced._meta.model_name)) + f"?id={fit.reduced.id}"

        return context
    


class FitDetailsView(SingleObjView):

    template_name = "iop4admin/view_fitdetails.html"

    def get_context_data(self, **kwargs):

        context = super().get_context_data(**kwargs)

        obj = self.get_object()

        # urls

        context['url_viewer'] = reverse(f"iop4admin:{self.model._meta.app_label}_{self.model._meta.model_name}_viewer", args=[obj.id])
        context['url_changelist'] = reverse('iop4admin:%s_%s_changelist' % (obj._meta.app_label, obj._meta.model_name)) + f"?id={obj.id}"

        if isinstance(obj, RawFit) and hasattr(obj, "reduced"):
            context['url_reduced'] = reverse('iop4admin:%s_%s_changelist' % (obj._meta.app_label, obj.reduced._meta.model_name)) + f"?id={obj.reduced.id}"

        if isinstance(obj, ReducedFit):
            context['url_raw'] = reverse('iop4admin:%s_%s_changelist' % (obj._meta.app_label, obj.rawfit._meta.model_name)) + f"?id={obj.rawfit.id}"

        ## non-empty fields and values
        fields_and_values = {field.name:str(getattr(obj, field.name)) for field in obj._meta.fields if getattr(obj, field.name) is not None}
        if hasattr(obj, 'flags'):
            fields_and_values['flags'] = ", ".join(obj.flag_labels)
        context['fields_and_values'] = fields_and_values

        # If file does not exist, return early

        if not obj.fileexists:
            return context
        
        # If file exists, gather additional info and pass it to the template
        
        imgdata = obj.mdata

        ## header

        with fits.open(obj.filepath) as hdul:
            context["header_L"] = [hdu.header for hdu in hdul]

        ## image preview parameters, image and histogram

        vmin = self.request.GET.get("vmin", "auto")
        vmax = self.request.GET.get("vmax", "auto")

        if vmin == "auto" or vmax == "auto":
            vmin = np.quantile(imgdata.compressed(), 0.3)
            vmax = np.quantile(imgdata.compressed(), 0.99)
            imgbytes = obj.get_imgbytes_preview_image()
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            imgbytes = obj.get_imgbytes_preview_image(vmin=vmin, vmax=vmax)

        imgb64 = base64.b64encode(imgbytes).decode("utf-8")

        context["vmin"] = vmin
        context["vmax"] = vmax
        context['imgb64'] = imgb64

        histimg = obj.get_imgbytes_preview_histogram()
        histimg_b64 =  base64.b64encode(histimg).decode("utf-8")
        context['histimg_b64'] = histimg_b64
        
        # Reduction info

        ## astrometric calibration
        try:
            context["astrometry_info"] = obj.astrometry_info
            # logger.debug(f"Loaded astrometry info for {obj}: {context['astrometry_info']}")
        except Exception as e:
            # logger.debug(f"Failed to load astrometry info for {obj}: {e}")
            pass

        #logger.debug(f"Loading astrometry images from disk for {obj.fileloc}")

        astrometry_img_D = dict()
        fnameL = glob.iglob(obj.filedpropdir + "/astrometry_*.png")
        for fname in fnameL:
            with open(fname, "rb") as f:
                imgbytes = f.read()
            imgb64 = base64.b64encode(imgbytes).decode("utf-8")
            astrometry_img_D[os.path.basename(fname)] = imgb64
        context["astrometry_img_D"] = dict(sorted(astrometry_img_D.items()))

        #logger.debug(f"Finished loading images from disk for {obj.fileloc}")

        ## sources info
        if isinstance(obj, ReducedFit):
            sources_in_field_L = list()
            for source in obj.sources_in_field.all():
                sources_in_field_L.append((source, reverse('iop4admin:%s_%s_changelist' % (source._meta.app_label, source._meta.model_name)) + f"?id={source.id}"))
            context["sources_in_field_L"] = sources_in_field_L


        # stats
        context["stats"] = obj.stats

        #logger.debug("Finished building template context for fit details view")

        return context
    
