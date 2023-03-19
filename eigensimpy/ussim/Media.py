
from eigensimpy.dsp.Signals import Signal

class IsotropicAcousticMedia:
    
    
    def __init__(self, **kwargs):
        self._comp_sos      = kwargs.get('comp_sos', Signal())
        self._shear_sos     = kwargs.get('shear_sos', Signal())
        self._density       = kwargs.get('density', Signal())
        self._comp_atten    = kwargs.get('comp_atten', Signal())
        self._shear_atten   = kwargs.get('shear_atten', Signal())
        
    @property
    def comp_sos(self):
        return self._comp_sos

    @comp_sos.setter
    def comp_sos(self, value):
        self._comp_sos = value

    @property
    def shear_sos(self):
        return self._shear_sos

    @shear_sos.setter
    def shear_sos(self, value):
        self._shear_sos = value

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        self._density = value

    @property
    def comp_atten(self):
        return self._comp_atten

    @comp_atten.setter
    def comp_atten(self, value):
        self._comp_atten = value

    @property
    def shear_atten(self):
        return self._shear_atten

    @shear_atten.setter
    def shear_atten(self, value):
        self._shear_atten = value