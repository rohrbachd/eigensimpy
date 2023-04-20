

from eigensimpy.ussim.Media import IsotropicElasticMedia
from eigensimpy.ussim.Recorders import RecorderSet2D
from eigensimpy.ussim.Boundaries import SimplePML
from eigensimpy.ussim.Transducers import ReceiverSet2D, EmitterSet2D
from eigensimpy.ussim.Fields import AcousticField2D
from eigensimpy.simmath.MathUtil import interp2
from eigensimpy.simmath.Derivatives import FirstOrderForward, FirstOrderBackward, MixedModelDerivative

import numpy as np

class SimSettings:
    def __init__(self, **kwargs):
        self.duration = kwargs.get('duration', [])
        self.duration_unit = kwargs.get('duration_unit', "")
        
    
class VirieuxDerivViscous2D:
    def __init__(self, **kwargs):
        self.media: IsotropicElasticMedia = kwargs.get('media', None)
        self.emitter: EmitterSet2D = kwargs.get('emitter', None)
        self.receiver: ReceiverSet2D = kwargs.get('receiver', None)
        self.settings: SimSettings = kwargs.get('settings', None)
        self.recorder: RecorderSet2D = kwargs.get('recorder', None)
        self.pml: SimplePML = kwargs.get('pml', None)
        

        self.dx1_fwrd: FirstOrderForward = FirstOrderForward(1)
        self.dx1_bwrd: FirstOrderBackward = FirstOrderBackward(1)

        self.dx2_fwrd: FirstOrderForward = FirstOrderForward(2)
        self.dx2_bwrd: FirstOrderBackward = FirstOrderBackward(2)

        self.dx1dt_fwrd: MixedModelDerivative = MixedModelDerivative(self.dx1_fwrd, FirstOrderForward(3))
        self.dx1dt_bwrd: MixedModelDerivative = MixedModelDerivative(self.dx1_bwrd, FirstOrderForward(3))

        self.dx2dt_fwrd: MixedModelDerivative = MixedModelDerivative(self.dx2_fwrd, FirstOrderForward(3))
        self.dx2dt_bwrd: MixedModelDerivative = MixedModelDerivative(self.dx2_bwrd, FirstOrderForward(3))
        
    def run_simulation(self):
        
        lamb = self.media.comp_modulus.data
        mu = self.media.shear_modulus.data 
        dens   = self.media.density.data

        eta = self.media.eta_atten.data
        chi = self.media.chi_atten.data

        field = AcousticField2D( field_size=mu.shape, dims=self.media.shear_modulus.dims)

        duration = self.settings.duration
        dt = self.emitter.delta

        n_sim_steps  = np.round_( duration / dt )

        time = np.arange(n_sim_steps ) * dt

        # we assume that delta 1 and delta 2 are the same
        delta_space  = self.media.shear_modulus.dims[0].delta
        
        B, Bi  = self.compute_buoyancy(dens, dt, delta_space)
        L2M, L = self.compute_lm_dim2( lamb, mu, dt, delta_space)
        Md1 = self.compute_m_dim1( mu, dt, delta_space )

        X2E, X = self.compute_xe_dim2( chi, eta, dt)
        E = self.compute_e_dim1( eta, dt )

        Bidtdx = Bi / dt / delta_space # deltaSpace is squared since Bi = 1/dens/dx
        Bdtdx = B / dt / delta_space # deltaSpace is squared since Bi = 1/dens/dx
            
        # get the raw data it should be faster to work with the raw
        # data directly
        vel1 = field.vel1.data
        vel2 = field.vel2.data

        stress11 = field.stress11.data
        stress22 = field.stress22.data
        stress12 = field.stress12.data

        recorder = self.recorder
        recorder.initialize(vel1)
        
        receiver = self.receiver
    
        # % si == sample index
        for si in range( int( np.round(n_sim_steps) )):
                
            ti = time[si];
            vel1 = self.emitter.emit_vel1(ti, vel1)
            vel2 = self.emitter.emit_vel2(ti, vel2)
            
            # first dimension z, or y commonly
            # Compute velocity and stress derivatives
            vel1, ds11d1, ds12d2 = self.compute_vel1(vel1, stress11, stress12, Bi)
            vel2, ds22d2, ds12d1 = self.compute_vel2(vel2, stress22, stress12, B)

            vel1 = self.pml.apply_pml( vel1, [True, True] ) # vel1 is staggered in d1 and d2 
            vel2 = self.pml.apply_pml( vel2, [False, False]) # vel2 is not staggered 
            
            ddv1dtd1 = self.dx1_bwrd.compute(ds11d1 + ds12d2) * Bidtdx
            ddv2dtd2 = self.dx2_fwrd.compute(ds22d2 + ds12d1) * Bdtdx
            ddv1dtd2 = self.dx2_bwrd.compute(ds11d1 + ds12d2) * Bidtdx
            ddv2dtd1 = self.dx1_fwrd.compute(ds22d2 + ds12d1) * Bdtdx

            if si > 481:    
                recorder.record_vel1(vel1)
                recorder.record_vel2(vel2)
                    
            receiver.record_vel1(ti, vel1)
            receiver.record_vel2(ti, vel2)

            stress11 = self.emitter.emit_stress11(ti, stress11)
            stress22 = self.emitter.emit_stress22(ti, stress22)
            stress12 = self.emitter.emit_stress12(ti, stress12)
            
            stress11 = self.compute_stress11(stress11, ddv2dtd2, ddv1dtd1, L2M, L, X2E, X, vel1, vel2)
            stress22 = self.compute_stress22(stress22, ddv2dtd2, ddv1dtd1, L2M, L, X2E, X, vel1, vel2)
            stress12 = self.compute_stress12(stress12, ddv2dtd1, ddv1dtd2, Md1, E, vel1, vel2)

            recorder.record_stress11(stress11)
            recorder.record_stress22(stress22)
            recorder.record_stress12(stress12)

            receiver.record_stress11(ti, stress11)
            receiver.record_stress22(ti, stress22)
            receiver.record_stress12(ti, stress12)
            

            
    def compute_vel1(self, vel1, stress11, stress12, Bi):
        
        ds11d1 = self.dx1_fwrd.compute(stress11)
        ds12d2 = self.dx2_fwrd.compute(stress12)

        vel1 = self.pml.apply_pml(vel1)

        vel1 = vel1 + Bi * ds11d1 + Bi * ds12d2

        velres = self.pml.apply_pml(vel1)

        return velres, ds11d1, ds12d2
    
    def compute_vel2(self, vel2, stress22, stress12, B):
        
        pml = self.pml

        ds22d2 = self.dx2_bwrd.compute(stress22)
        ds12d1 = self.dx1_bwrd.compute(stress12)

        vel2 = pml.apply_pml(vel2)

        vel2 = vel2 + B * ds22d2 + B * ds12d1

        velres = pml.apply_pml(vel2)

        return velres, ds22d2, ds12d1
    
    def compute_stress11(self, stress11, dduydydt, dduxdxdt, L2M, L, X2E, X, vel1, vel2):
        
        pml = self.pml

        stress11 = pml.apply_pml(stress11)

        stress11 = stress11 + L2M * self.dx1_bwrd.compute(vel1) \
                            + X2E * dduxdxdt \
                            + L * self.dx2_fwrd.compute(vel2) \
                            + X * dduydydt

        stress11 = pml.apply_pml(stress11)

        return stress11
    
    def compute_stress22(self, stress22, dduydydt, dduxdxdt, L2M, L, X2E, X, vel1, vel2):
        pml = self.pml

        stress22 = pml.apply_pml(stress22)

        stress22 = stress22 + L * self.dx1_bwrd.compute(vel1) \
                            + X * dduxdxdt \
                            + L2M * self.dx2_fwrd.compute(vel2) \
                            + X2E * dduydydt

        stress22 = pml.apply_pml(stress22)

        return stress22
    
    def compute_stress12(self, stress12, dduydxdt, dduxdydt, M, E, vel1, vel2):
        pml = self.pml

        stress12 = pml.apply_pml(stress12)

        stress12 = stress12 + M * self.dx1_fwrd.compute(vel2) \
                            + E * dduydxdt \
                            + M * self.dx2_bwrd.compute(vel1) \
                            + E * dduxdydt

        stress12 = pml.apply_pml(stress12)

        return stress12
    
    
    def compute_xe_dim2(self, chi, eta, dt):
        x2e = chi + 2 * eta

        xq, yq = np.meshgrid(np.arange(1.5, x2e.shape[1] + 1.5), np.arange(1, x2e.shape[0] + 1))

        X2E = interp2(x2e, xq, yq)

        x2e_nan = np.isnan(X2E)
        X2E[x2e_nan] = x2e[x2e_nan]

        xq, yq = np.meshgrid(np.arange(1.5, chi.shape[1] + 1.5), np.arange(1, chi.shape[0] + 1))

        X = interp2(chi, xq, yq)
        x_nan = np.isnan(X)
        X[x_nan] = chi[x_nan]

        X2E = X2E * dt
        X = X * dt

        return X2E, X


    def compute_e_dim1(self, eta, dt):
        xq, yq = np.meshgrid(np.arange(1, eta.shape[1] + 1), np.arange(1.5, eta.shape[0] + 1.5))

        Ed1 = interp2(eta, xq, yq)

        e_nan = np.isnan(Ed1)
        Ed1[e_nan] = eta[e_nan]

        Ed1 = Ed1 * dt

        return Ed1
    
    def compute_lm_dim2(self, lambda_, mu, dt, delta_space):
        l2m = lambda_ + 2 * mu

        xq, yq = np.meshgrid(np.arange(1.5, l2m.shape[1] + 1.5), np.arange(1, l2m.shape[0] + 1))

        L2M = interp2(l2m, xq, yq)

        l2m_nan = np.isnan(L2M)
        L2M[l2m_nan] = l2m[l2m_nan]
        L2M = L2M * dt / delta_space

        xq, yq = np.meshgrid(np.arange(1.5, lambda_.shape[1] + 1.5), np.arange(1, lambda_.shape[0] + 1))

        L = interp2(lambda_, xq, yq)

        l_nan = np.isnan(L)
        L[l_nan] = lambda_[l_nan]
        L = L * dt / delta_space

        return L2M, L

    def compute_m_dim1(self, mu, dt, delta_space):
        xq, yq = np.meshgrid(np.arange(1, mu.shape[1] + 1), np.arange(1.5, mu.shape[0] + 1.5))

        Md1 = interp2(mu, xq, yq)

        m_nan = np.isnan(Md1)
        Md1[m_nan] = mu[m_nan]

        Md1 = Md1 * dt / delta_space

        return Md1

    def compute_buoyancy(self, dens, dt, delta_space):
        B = (1 / dens) * dt / delta_space

        xq, yq = np.meshgrid(np.arange(1.5, B.shape[1] + 1.5), np.arange(1.5, B.shape[0] + 1.5))

        Bi = interp2(B, xq, yq)

        bi_nan = np.isnan(Bi)
        Bi[bi_nan] = 0 #B[bi_nan]

        return B, Bi