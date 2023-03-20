
from eigensimpy.dsp.Signals import Signal
import numpy as np

class IsotropicElasticMedia:
    
    def __init__(self, **kwargs):
        self._comp_modulus    = kwargs.get('comp_modulus', Signal())
        self._shear_modulus   = kwargs.get('shear_modulus', Signal())
        self._density         = kwargs.get('density', Signal())
        self._eta_atten       = kwargs.get('eta_atten', Signal())
        self._chi_atten       = kwargs.get('chi_atten', Signal())

    @property
    def comp_modulus(self):
        return self._comp_modulus

    @comp_modulus.setter
    def comp_modulus(self, value):
        self._comp_modulus = value

    @property
    def shear_modulus(self):
        return self._shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, value):
        self._shear_modulus = value

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        self._density = value

    @property
    def eta_atten(self):
        return self._eta_atten

    @eta_atten.setter
    def eta_atten(self, value):
        self._eta_atten = value

    @property
    def chi_atten(self):
        return self._chi_atten

    @chi_atten.setter
    def chi_atten(self, value):
        self._chi_atten = value

class IsotropicAcousticMedia:
    
    def __init__(self, **kwargs):
        self._comp_sos      = kwargs.get('comp_sos', Signal())
        self._shear_sos     = kwargs.get('shear_sos', Signal())
        self._density       = kwargs.get('density', Signal())
        self._comp_atten    = kwargs.get('comp_atten', Signal())
        self._shear_atten   = kwargs.get('shear_atten', Signal())
        # for now we use power of two
        self._power = 2
    
    def convert_to_elastic(self):
        density = self._density
        mu = self._shear_sos**2 * self._density
        lambda_ = self._comp_sos**2 * self._density - 2 * mu

        eta = self.compute_eta()
        chi = self.compute_chi(eta)

        lame = IsotropicElasticMedia(comp_modulus=lambda_, shear_modulus=mu, density=density, eta_atten=eta, chi_atten=chi)
        return lame
    
    def compute_chi(self, eta=None):
        if eta is None:
            eta = self.compute_eta()

        r = self._density
        cp = self._comp_sos
        y = self._power
        ap = self._comp_atten

        chi = 2 * r * cp**3 * self.db2neper(ap, y) - 2 * eta
        return chi

    def compute_eta(self):
        r = self._density
        cs = self._shear_sos
        y = self._power
        as_ = self._shear_atten

        eta = 2 * r * cs**3 * self.db2neper(as_, y)
        return eta

    

    @staticmethod
    def db2neper(atten_db_mhz_cm, power):
        neper = 8.6858896380650366
        cm = 100
        scale = cm * (1e-6 / (2 * np.pi))**power
        atten_np = atten_db_mhz_cm * scale / neper
        return atten_np
        
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