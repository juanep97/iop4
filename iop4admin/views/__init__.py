from iop4lib.db import *
from .singleobj import *
from .fitfile import *


class AstroSourceDetailsView(SingleObjView):
    model = AstroSource
    template_name = "iop4admin/view_astrosourcedetails.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        obj = self.get_object()

        fields_and_values = {field.name:field.value_to_string(obj) for field in AstroSource._meta.fields if field.name != "comment" and getattr(obj, field.name) is not None}
        context['fields_and_values'] = fields_and_values

        return context
    

class EpochDetailsView(SingleObjView):
    model = Epoch
    template_name = "iop4admin/view_epochdetails.html"

    def get_context_data(self, **kwargs):
        import itertools

        context = super().get_context_data(**kwargs)

        obj = self.get_object()

        try:
            header_key_S = set(itertools.chain.from_iterable([rawfit.header.keys() for rawfit in obj.rawfits.all()]))
            context["header_key_S"] = list(header_key_S)
        except Exception as e:
            pass

        #header_key_D = dict()
        #for key in header_key_S:
        #    header_key_D[key] = len([rawfit.id for rawfit in obj.rawfits.all() if key in rawfit.header])
        

        context["rawfitsummarystatus"] = obj.get_summary_rawfits_status()

        return context