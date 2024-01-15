# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# other imports
import os
from pathlib import Path
import io
import base64
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt

# iop4lib
from iop4lib.db import AstroSource
from .singleobj import SingleObjView
from iop4lib.utils.plotting import plot_finding_chart

# logging
import logging
logger = logging.getLogger(__name__)

class AstroSourceDetailsView(SingleObjView):
    model = AstroSource
    template_name = "iop4admin/view_astrosourcedetails.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        obj = self.get_object()

        fields_and_values = {field.name:field.value_to_string(obj) for field in AstroSource._meta.fields if field.name != "comment" and getattr(obj, field.name) is not None}
        context['fields_and_values'] = fields_and_values

        # finding chart
        finding_char_path = Path(obj.filedpropdir) / "finding_chart.png"

        if not os.path.exists(finding_char_path) or iop4conf.iop4admin['force_rebuild_finding_charts']:
            buf = io.BytesIO()

            width, height = 800, 800

            fig = mplt.figure.Figure(figsize=(width/100, height/100), dpi=iop4conf.mplt_default_dpi)
            ax = fig.subplots()

            plot_finding_chart(obj, ax=ax, fig=fig)

            fig.savefig(buf, format='png', bbox_inches='tight')
            fig.clf()

            buf.seek(0)
            imgbytes = buf.read()

            if not os.path.exists(obj.filedpropdir):
                os.makedirs(obj.filedpropdir)
            with open(finding_char_path, 'wb') as f:
                f.write(imgbytes)
        else: 
            with open(finding_char_path, 'rb') as f:
                imgbytes = f.read()

        context['finding_chart_b64'] = base64.b64encode(imgbytes).decode('utf-8')

        return context
    
