import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

import os
import dill
from functools import wraps

import logging
logger = logging.getLogger(__name__)

class FiledProperty():
    """ An instance attributed stored in the .filedpropdir of the instance as a pickle file. """
    
    def __init__(self, fget=None, fset=None, name=None, readonly=False):  
        self.readonly = readonly
        self.fget = fget
        self.fset = fset
        self.name = name

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name

        self.owner = owner

    def setter(self, func):
        prop = type(self)(fget=self.fget, fset=func, name=self.name)
        return prop
        
    def getter(self, func):
        prop = type(self)(fget=func, fset=self.fset, name=self.name)
        return prop

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
            
        if self.fget is None:
            return self._fget_default(obj)
        else:
            return self.fget(obj)
        
    def __set__(self, obj, value):  
        if obj is None:
            return self
            
        if self.fset is None:
            if self.readonly:
                raise AttributeError("Can not set attribute")
            else:
                return self._fset_default(obj, value)
        else:
            return self.fset(obj, value)

    def _get_prop_path(self, obj):
        if not hasattr(obj, 'filedpropdir'):
            raise AttributeError("Object has no filedpropdir attribute")
        
        if not os.path.isdir(obj.filedpropdir):
            os.makedirs(obj.filedpropdir)

        return os.path.join(obj.filedpropdir, self.name)
    
    def _fget_default(self, obj):
        prop_path = self._get_prop_path(obj)
        if not os.path.isfile(prop_path):
            raise NameError
        with open(prop_path, 'rb') as f:
            return dill.load(f)
        
    def _fset_default(self, obj, value):
        prop_path = self._get_prop_path(obj)
        with open(prop_path, 'wb') as f:
            dill.dump(value, f)
            

def filed_property(name=None, readonly=True):
    
    def decorator(fget):
        if name is None:
            return FiledProperty(name=fget.__name__, readonly=readonly, fget=fget)
        if name is not None:
            return FiledProperty(name=name, readonly=readonly, fget=fget)
        
    return decorator