from django.db import models

from django.db.models import Lookup

from enum import Enum

class FlagChoices(Enum):

    def __add__(self, other):
        if isinstance(other, self.__class__):
            otherval = other.value
        else:
            otherval = other

        return self.value | otherval

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            otherval = other.value
        else:
            otherval = other

        if self.value & otherval != 0:
            return self.value & ~otherval
        else:
            return otherval
        
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    @classmethod
    def choices(cls):
        return [(flag.value, flag.name) for flag in cls]

    @classmethod
    def get_label(cls, flag_value):
        """ Returns the label for the given flag value, does NOT allow for combinations of flags. 
        """
        if flag_value == 0:
            return '-'
        return cls(flag_value).name

    @classmethod
    def get_labels(cls, flag_value):
        """ Returns a list of labels for the given flag value, allows for combinations of flags. 
        """
        return [
            label
            for value, label in cls.choices()
            if value & flag_value != 0]


class FlagLookup(Lookup):
    """
    Any Lookup for this field should subclass this so it can accept the Enum directly.
    """
    def process_rhs(self, compiler, connection):
        if isinstance(self.rhs, Enum):
            self.rhs = self.rhs.value
        return super().process_rhs(compiler, connection)
    
class HasFlagLookup(FlagLookup):
    lookup_name = "has"

    def as_sql(self, compiler, connection):
        lhs, lhs_params = self.process_lhs(compiler, connection)
        rhs, rhs_params = self.process_rhs(compiler, connection)
        params = lhs_params + rhs_params
        return "%s & %s" % (lhs, rhs), params
    
class HasNotFlagLookup(FlagLookup):
    lookup_name = "hasnot"

    def as_sql(self, compiler, connection):
        lhs, lhs_params = self.process_lhs(compiler, connection)
        rhs, rhs_params = self.process_rhs(compiler, connection)
        params = lhs_params + rhs_params
        return "NOT (%s & %s)" % (lhs, rhs), params
    
class FlagPropertyDescriptor:
    def __init__(self, field):
        self.field = field

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.field.attname]

    def __set__(self, instance, value):
        """ Allows to set it directly with a FlagChoice.FLAGNAME """
        if isinstance(value, Enum):
            value = value.value
        instance.__dict__[self.field.attname] = value

class FlagBitField(models.BigIntegerField):
    def get_prep_value(self, value):
        if isinstance(value, Enum):
            return value.value
        return super().get_prep_value(value)
    
    def __init__(self, choices=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.choices = choices
        self.register_lookup(HasFlagLookup)
        self.register_lookup(HasNotFlagLookup)

    def contribute_to_class(self, cls, name, **kwargs):
        """ 
        Provides methods to set, unset and check flags
        and give the object properties to retrieve flags as enum, 
        list of values and list of labels.
        """

        super().contribute_to_class(cls, name, **kwargs)
        setattr(cls, self.name, FlagPropertyDescriptor(self))

        def clear_flags(instance):
            instance.flags = 0
            return None
        
        def set_flag(instance, flag):
            if isinstance(flag, Enum):
                value = flag.value
            else:
                value = flag
            instance.flags |= value
            return None

        def unset_flag(instance, flag):
            if isinstance(flag, Enum):
                value = flag.value
            else:
                value = flag
            instance.flags &= ~value
            return None

        def has_flag(instance, flag):
            if isinstance(flag, Enum):
                value = flag.value
            else:
                value = flag
            return instance.flags & value != 0
        
        def has_not_flag(instance, flag):
            if isinstance(flag, Enum):
                value = flag.value
            else:
                value = flag
            return not (instance.flags & value != 0)
    
        @property
        def flag_list(instance):
            """ List of flags set, as enum choices (value, label). 
            """
            return [
                choice
                for choice in self.choices
                if instance.has_flag(choice[0])
            ]
        
        @property
        def flag_values(instance):
            """ list of values for the flags set. 
            """
            return [
                value
                for value, label in self.choices
                if instance.has_flag(value)
            ]
        
        @property
        def flag_labels(instance):
            """ List of labels for the flags set. 
            """
            return [
                label
                for value, label in self.choices
                if instance.has_flag(value)
            ]
        
        # methods for flag managing

        cls.clear_flags = clear_flags
        cls.set_flag = set_flag
        cls.unset_flag = unset_flag
        cls.has_flag = has_flag
        cls.has_not_flag = has_not_flag

        # properties for flag retrieving

        cls.flag_list = flag_list
        cls.flag_values = flag_values
        cls.flag_labels = flag_labels

        # method for multiple flag checking
        
        def has_flags(instance, flags):
            """ Return True if all flags given are set. 
            """
            return all([instance.has_flag(flag) for flag in flags])
        
        def has_not_flags(instance, flags):
            """ Return True if all flags given are NOT set.
            """
            return all([instance.has_not_flag(flag) for flag in flags])
    
        cls.has_flags = has_flags
        cls.has_not_flags = has_not_flags
        
