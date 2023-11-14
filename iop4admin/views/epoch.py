# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# other imports

import itertools

# iop4lib

from iop4lib.db import Epoch
from .singleobj import SingleObjView

# logging

import logging
logger = logging.getLogger(__name__)

class EpochDetailsView(SingleObjView):
    model = Epoch
    template_name = "iop4admin/view_epochdetails.html"

    def get_context_data(self, **kwargs):

        context = super().get_context_data(**kwargs)

        obj = self.get_object()

        try:
            header_key_S = set(itertools.chain.from_iterable([rawfit.header.keys() for rawfit in obj.rawfits.all()]))
            context["header_key_S"] = list(header_key_S)
        except Exception as e:
            pass

        context["rawfitsummarystatus"] = obj.get_summary_rawfits_status()

        return context