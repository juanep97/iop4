from django.shortcuts import render, get_object_or_404

from ..models import *

# other imports

# logging

import logging
logger = logging.getLogger(__name__)

#@staff_member_required
# no staff_member_required, because it is only routed from iop4admin_site
def view_astrosourcedetails(site, request, astrosource_id):  

    context = dict(site.each_context(request))

    obj = get_object_or_404(AstroSource, id=astrosource_id)
    #astrosrc = AstroSource.objects.filter(id=astrosource_id).first()

    context["title"] = f"Source {obj.name}"
    
    context['astrosrc'] = obj

    #fields_and_values = [(field.name, field.value_to_string(obj)) for field in AstroSource._meta.fields if field.name != "comment"]
    fields_and_values = [(field.name, field.value_to_string(obj)) for field in AstroSource._meta.fields if field.name != "comment" and getattr(obj, field.name) is not None]
    context['fields_and_values'] = fields_and_values

    context['comment_html'] = obj.comment_html

    return  render(request, "iop4admin/view_astrosourcedetails.html", context)