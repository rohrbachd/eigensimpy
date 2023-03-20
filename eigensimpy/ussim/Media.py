
from eigensimpy.dsp.Signals import Signal
import eigensimpy.dsp.Operations as ops
import numpy as np

NEPER = 8.6858896380650366

class IsotropicElasticMedia:
    
    def __init__(self, **kwargs):
        self._comp_modulus    = kwargs.get('comp_modulus', Signal())
        self._shear_modulus   = kwargs.get('shear_modulus', Signal())
        self._density         = kwargs.get('density', Signal())
        self._eta_atten       = kwargs.get('eta_atten', Signal())
        self._chi_atten       = kwargs.get('chi_atten', Signal())

    def convert_to_acoustic(self):
        density = self._density
        lambda_ = self._comp_modulus
        mu = self._shear_modulus

        # Calculate compressional and shear speeds of sound
        comp_sos = ops.sqrt((lambda_ + 2 * mu) / density)
        shear_sos = ops.sqrt(mu / density)

        # Calculate power-law exponent (assuming the same for both compressional and shear waves)
        y = 1  # Assuming power-law exponent is 1, you might want to adjust this based on your application

        # Calculate compressional wave attenuation from chi_atten
        ap = self.neper2db( self._chi_atten / (2 * density * comp_sos ** 3) + self._eta_atten / (2 * density * shear_sos ** 3), y)

        acoustic_media = IsotropicAcousticMedia(comp_sos=comp_sos, 
                                                shear_sos=shear_sos, 
                                                density=density, 
                                                comp_atten=ap, 
                                                shear_atten=None)
        return acoustic_media

    @staticmethod
    def neper2db(atten_np, power):
        #neper = 8.6858896380650366
        cm = 100
        scale = cm * (1e-6 / (2 * np.pi)) ** power
        atten_db_mhz_cm = atten_np * NEPER / scale
        return atten_db_mhz_cm
     
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
        
        
        def_shape = kwargs.get('default_shape', (0,0) )
        
        if len(kwargs) > 0:
            first_signal = next(iter(kwargs.values()))
            if isinstance(first_signal, Signal):
                def_shape = first_signal.shape    
        
        self._comp_sos      = kwargs.get('comp_sos',    Signal( data=np.zeros(def_shape )))
        self._shear_sos     = kwargs.get('shear_sos',   Signal( data=np.zeros(def_shape )))
        self._density       = kwargs.get('density',     Signal( data=np.zeros(def_shape )))
        self._comp_atten    = kwargs.get('comp_atten',  Signal( data=np.zeros(def_shape )))
        self._shear_atten   = kwargs.get('shear_atten', Signal( data=np.zeros(def_shape )))
        
        # for now we use power of two
        self._power = 2
        
        sosShape = self._comp_sos.shape
        # make sure all Signal classes have the same shape
        shapes = [ self._comp_sos.shape, 
                   self._shear_sos.shape, 
                   self._density.shape, 
                   self._comp_atten.shape, 
                   self._shear_atten.shape ]
            
        for s in shapes:
            if sosShape != s :
                raise ValueError("all input materials must have equal shape") 
            
            
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
        #neper = 8.6858896380650366
        cm = 100
        scale = cm * (1e-6 / (2 * np.pi))**power
        atten_np = atten_db_mhz_cm * scale / NEPER
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