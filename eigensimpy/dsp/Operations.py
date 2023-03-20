import numpy as np
from eigensimpy.dsp.Signals import Signal


def sqrt(sig):    
    data = np.sqrt(sig.data)
    alternative = lambda x : np.sqrt(x)
    return _compute_signal(sig, data, alternative)


def cos(sig):
    data = np.cos(sig.data)
    alternative = lambda x : np.cos(x)
    return _compute_signal(sig, data, alternative)


def sin(sig):
    data = np.sin(sig.data)
    alternative = lambda x : np.sin(x)
    return _compute_signal(sig, data, alternative)


def log(sig):
    data = np.log(sig.data)
    alternative = lambda x : np.log(x)
    return _compute_signal(sig, data, alternative)


def log10(sig):
    data = np.log10(sig.data)
    alternative = lambda x : np.log10(x)
    return _compute_signal(sig, data, alternative)
    
    
def arcsin(sig):
    data = np.arcsin(sig.data)
    alternative = lambda x : np.arcsin(x)
    return _compute_signal(sig, data, alternative)


def arccos(sig):
    data = np.arccos(sig.data)
    alternative = lambda x : np.arccos(x)
    return _compute_signal(sig, data, alternative)


def tan(sig):
    data = np.tan(sig.data)
    alternative = lambda x : np.tan(x)
    return _compute_signal(sig, data, alternative)


def arctan(sig):
    data = np.arctan(sig.data)
    alternative = lambda x : np.arctan(x)
    return _compute_signal(sig, data, alternative)

def _compute_signal(sig, data, alternative):
    if isinstance(sig, Signal):
        return Signal(data=data, dims=sig.dims.copy(), amplitude=sig.amplitude)
    elif isinstance(sig, np.ndarray):
        return alternative(sig)
    else:
        raise TypeError(f"Unsupported input type {type(sig)}")
