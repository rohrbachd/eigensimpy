

from eigensimpy.ussim.Media import IsotropicElasticMedia
from eigensimpy.ussim.Recorders import RecorderSet2D
from eigensimpy.ussim.Boundaries import SimplePML
from eigensimpy.ussim.Transducers import ReceiverSet2D, EmitterSet2D

from eigensimpy.simmath.Derivatives import FirstOrderForward, FirstOrderBackward, MixedModel

class SimSettings:
    def __init__(self, **kwargs):
        self.duration = kwargs.get('duration', [])
        self.duration_unit = kwargs.get('duration_unit', "")
        
    
class VirieuxDerivViscous2D:
    def __init__(self, **kwargs):
        self.media: IsotropicElasticMedia = kwargs.get('media', None)
        self.emitter: EmitterSet2D = kwargs.get('emitter', None)
        self.settings: SimSettings = kwargs.get('settings', None)
        self.recorder: RecorderSet2D = kwargs.get('recorder', None)
        self.pml: SimplePML = kwargs.get('pml', SimplePML())
        self.receiver: ReceiverSet2D = kwargs.get('receiver', None)

        self.dx1_fwrd: FirstOrderForward = FirstOrderForward(1)
        self.dx1_bwrd: FirstOrderBackward = FirstOrderBackward(1)

        self.dx2_fwrd: FirstOrderForward = FirstOrderForward(2)
        self.dx2_bwrd: FirstOrderBackward = FirstOrderBackward(2)

        self.dx1dt_fwrd: MixedModel = MixedModel(self.dx1_fwrd, FirstOrderForward(3))
        self.dx1dt_bwrd: MixedModel = MixedModel(self.dx1_bwrd, FirstOrderForward(3))

        self.dx2dt_fwrd: MixedModel = MixedModel(self.dx2_fwrd, FirstOrderForward(3))
        self.dx2dt_bwrd: MixedModel = MixedModel(self.dx2_bwrd, FirstOrderForward(3))
        
    def runSimulation(self):
        
        lambda = self.media.comp_modulus.data
        mu = self.media.shear_modulus.data 
        
        dens   = self.media.density.data

        eta = self.media.eta_atten.data;
        chi = self.media.chi_atten.data;

        field = ussim.fields.AcousticField2D(size(mu), self.media.ShearModulus.Dimensions);

        duration = self.Settings.Duration;
        dt = self.Emitter.Delta;

        nSimSteps = round( duration ./ dt ); 

        time = ( 0:nSimSteps-1 ) .* dt;

        # % we assume that delta 1 and delta 2 are the same
        # deltaSpace = self.media.ShearModulus.Dimensions(1).Delta;
        
        # [B, Bi]   = computeBuoyancy(dens, dt, deltaSpace);
        # [ L2M, L] = computeLMdim2( lambda, mu, dt, deltaSpace);
        # Md1 = computeMdim1( mu, dt, deltaSpace );


        # [ X2E, X] = computeXEdim2( chi, eta, deltaSpace);
        # E = computeEd im1( eta, deltaSpace );


        # % get the raw data it should be faster to work with the raw
        # % data directly
        # vel1 = field.Vel1.Data;
        # vel2 = field.Vel2.Data;

        # vel1 = cat(3,vel1,vel1);
        # vel2 = cat(3,vel2,vel2);

        # stress11 = field.Stress11.Data;
        # stress22 = field.Stress22.Data;
        # stress12 = field.Stress12.Data;

        # recorder = self.Recorder;
        # recorder.initialize(field);

        # receiver = self.Receiver;
    
        # pml = self.Pml;

        # % si == sample index
        # for si = 1:nSimSteps
                
        #     ti = time(si);
        #     vel1(:,:,2) = self.Emitter.emittVel1(ti, vel1(:,:,2));
        #     vel2(:,:,2) = self.Emitter.emittVel2(ti, vel2(:,:,2));
            
        #     % first dimension z, or y commonly
        #     vel1 = self.computeVel1(vel1, stress11 , stress12, Bi);
        #     vel2 = self.computeVel2(vel2, stress22, stress12, B );

        #     vel1(:,:,2) = pml.applyPml( vel1(:,:,2), [true true]); % vel1 is staggered in d1 and d2 
        #     vel2(:,:,2) = pml.applyPml( vel2(:,:,2), [false false]); % vel2 is not staggered 
            
        #     recorder.RecorderVel1.recordField(vel1(:,:,2));
        #     recorder.RecorderVel2.recordField(vel2(:,:,2));
            
        #     receiver.receiveVel1(ti, vel1(:,:,2));
        #     receiver.receiveVel2(ti, vel2(:,:,2));

        #     stress11 = self.Emitter.emittStress11(ti, stress11);
        #     stress22 = self.Emitter.emittStress22(ti, stress22);
        #     stress12 = self.Emitter.emittStress12(ti, stress12);

        #     stress11 = self.computeStress11(stress11, vel1, vel2, L2M, L, X2E, X);
        #     stress22 = self.computeStress22(stress22, vel1, vel2, L2M, L, X2E, X);
        #     stress12 = self.computeStress12(stress12, vel1, vel2, Md1, E);

        #     stress11 = pml.applyPml( stress11, [false true]); % vel2 is not staggered 
        #     stress22 = pml.applyPml( stress22, [true false]); % vel2 is not staggered 
        #     stress12 = pml.applyPml( stress12, [true false]); % vel2 is not staggered 

        #     recorder.RecorderStress11.recordField(stress11);
        #     recorder.RecorderStress11.recordField(stress22);
        #     recorder.RecorderStress11.recordField(stress12);
            
        #     receiver.receiveStress11(ti, stress11);
        #     receiver.receiveStress22(ti, stress22);
        #     receiver.receiveStress12(ti, stress12);
            
       