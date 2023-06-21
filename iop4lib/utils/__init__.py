from .filedproperty import *
from .plotting import *
from .sourcedetection import *
from .sourcepairing import *
from .astrometry import *

import os
import psutil

def divisorGenerator(n):
    """Generator for divisors of n"""
    large_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

def stats_dict(data):
        import numpy as np
        import scipy as sp
        import scipy.stats as stats

        if isinstance(data, np.ma.MaskedArray):
            data = data.compressed()
        else:
              data = data.flatten()

        res = dict()

        res['mean'] = np.mean(data)
        res['median'] = np.median(data)
        res['std'] = np.std(data)
        res['min'] = np.min(data)
        res['max'] = np.max(data)
        res['mode'] = stats.mode(data, axis=None, keepdims=False).mode

        return res

def get_mem_children():
    children = psutil.Process(os.getpid()).children(recursive=True)
    memory_usage = 0
    for child in children:
        try:
            mem_info = child.memory_info()
            memory_usage += mem_info.rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return memory_usage

def get_mem_current():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def get_mem_parent_from_child():
    process = psutil.Process(os.getpid()).parent()
    mem_info = process.memory_info()
    return mem_info.rss

def get_mem_children_from_child():
    parent = psutil.Process(os.getpid()).parent()
    children = parent.children(recursive=True)
    memory_usage = 0
    for child in children:
        try:
            mem_info = child.memory_info()
            memory_usage += mem_info.rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return memory_usage

def get_total_mem_from_child():
    return get_mem_parent_from_child() + get_mem_children_from_child()




