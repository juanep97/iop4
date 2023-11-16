from django.contrib import admin

# logging
import logging
logger = logging.getLogger(__name__)

class TextInputFilter(admin.SimpleListFilter):
    """
    Generic class for an input filter.
    """
    template = 'iop4admin/fits/textinput_filter.html'

    def lookups(self, request, model_admin):
        # Dummy, required to show the filter.
        return ((),)

    def choices(self, changelist):
        # Grab only the "all" option.
        all_choice = next(super().choices(changelist))
        all_choice['query_parts'] = (
            (k, v)
            for k, v in changelist.get_filters_params().items()
            if k != self.parameter_name
        )
        yield all_choice

class RawFitIdFilter(TextInputFilter):
    title = 'id'
    name = 'id'
    parameter_name = 'id'

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(id=self.value())
        
class RawFitNightFilter(TextInputFilter):
    title = 'Night'
    name = 'night'
    parameter_name = 'night'

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(epoch__night=self.value())
        
class RawFitFilenameFilter(TextInputFilter):
    title = 'Filename (regex)'
    name = 'filename'
    parameter_name = 'filename'

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(filename__iregex=self.value())

class RawFitTelescopeFilter(admin.SimpleListFilter):
    """
    Filter list for telescope in RawFit.
    """
    title = 'telescope'
    name = 'telescope'
    parameter_name = 'telescope'

    def lookups(self, request, model_admin):
        from iop4lib.telescopes import Telescope
        return ((t.name, t.name) for t in Telescope.get_known())

    def queryset(self, request, queryset):
        from iop4lib.telescopes import Telescope
        if (val := self.value()) is not None:
            return queryset.filter(epoch__telescope=Telescope.by_name(val).name)
        
class RawFitInstrumentFilter(admin.SimpleListFilter):
    """
    Filter list for instrument in RawFit.
    """
    title = 'instrument'
    name = 'instrument'
    parameter_name = 'instrument'

    def lookups(self, request, model_admin):
        from iop4lib.instruments import Instrument
        return ((i.name, i.name) for i in Instrument.get_known())
    
    def queryset(self, request, queryset):
        from iop4lib.instruments import Instrument
        if (val := self.value()) is not None:
            return queryset.filter(instrument=Instrument.by_name(val).name)

class RawFitFlagFilter(admin.SimpleListFilter):

    title = 'flags'
    name = 'flags'
    parameter_name = 'flags'

    def __init__(self, request, params, model, model_admin):
        super().__init__(request, params, model, model_admin)
        self.request = request
        self.model = model
        self.model_admin = model_admin

    def lookups(self, request, model_admin):
        from iop4lib.db import RawFit
        return ((flag.name, flag.value) for flag in RawFit.FLAGS)


    def queryset(self, request, queryset):
        if self.value() is None:
            return queryset
        
        selected_values = self.value().split(',')

        # Get the list of selected values
        selected_values = self.value().split(',')

        if "OPT_EXACT" in selected_values:
            OPT_EXACT = True
            selected_values.remove("OPT_EXACT")
        else:
            OPT_EXACT = False 


        if "OPT_NOTEXACT" in selected_values:
            OPT_NOTEXACT = True
            selected_values.remove("OPT_NOTEXACT")
        else:
            OPT_NOTEXACT = False

        if len(selected_values) == 0:
            return queryset

        # Filter objects using the custom method
        filtered_objects = []
        for obj in queryset:
            flags_yes= [self.model.FLAGS[val] for val in selected_values]
            flags_no = list(set(self.model.FLAGS)-set(flags_yes))

            flags_yes = [flag.value for flag in flags_yes]
            flags_no = [flag.value for flag in flags_no]

            if OPT_EXACT:
                if obj.has_flags(flags_yes) and obj.has_not_flags(flags_no):
                    filtered_objects.append(obj.id)
            elif OPT_NOTEXACT:
                if obj.has_flags(flags_yes):
                    filtered_objects.append(obj.id)

        return queryset.filter(id__in=filtered_objects)

    def choices(self, changelist):
        selected_values = self.value().split(',') if self.value() is not None else []

        # give default value
        if "OPT_EXACT" not in selected_values and "OPT_NOTEXACT" not in selected_values:
            selected_values.append("OPT_NOTEXACT")
        
        opt_exact_selected = "OPT_EXACT" in selected_values
        opt_notexact_selected = "OPT_NOTEXACT" in selected_values

        # AND OPTION
        new_values = selected_values.copy()

        if opt_exact_selected:
            new_values.remove("OPT_EXACT")
        else:
            new_values.append("OPT_EXACT")
            if opt_notexact_selected:
                new_values.remove("OPT_NOTEXACT")

        if len(new_values) == 0:
            query_string = changelist.get_query_string(remove=[self.parameter_name])
        else:
            query_string = changelist.get_query_string({self.parameter_name: ','.join(new_values)})

        yield {
            'selected': opt_exact_selected,
            'query_string': query_string,
            'display': 'exactly ' + ('(X)' if opt_exact_selected else '( )'),
        }

        # OR OPTION
        new_values = selected_values.copy()

        if opt_notexact_selected:
            new_values.remove("OPT_NOTEXACT")
        else:
            new_values.append("OPT_NOTEXACT")
            if opt_exact_selected:
                new_values.remove("OPT_EXACT")

        if len(new_values) == 0:
            query_string = changelist.get_query_string(remove=[self.parameter_name])
        else:
            query_string = changelist.get_query_string({self.parameter_name: ','.join(new_values)})
                
        yield {
            'selected': opt_notexact_selected,
            'query_string': query_string,
            'display': 'at least ' + ('(X)' if opt_notexact_selected else '( )'),
        }

        # VALUES 

        for lookup, title in self.lookup_choices:
            new_values = selected_values.copy()

            is_selected = lookup in selected_values

            if is_selected:
                new_values.remove(lookup)
            else:
                new_values.append(lookup)

            if len(new_values) == 0:
                query_string = changelist.get_query_string(remove=[self.parameter_name])
            else:
                query_string = changelist.get_query_string({self.parameter_name: ','.join(new_values)})
                
            yield {
                'selected': is_selected,
                'query_string':  query_string,
                'display': lookup, # title
            }

