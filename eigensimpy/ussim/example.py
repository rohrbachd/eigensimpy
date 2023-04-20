from eigensimpy.ussim.Materials import ViscousMaterial
from eigensimpy.ussim.Solvers import SimSettings, VirieuxDerivViscous2D
from eigensimpy.ussim.Recorders import FieldDisplay2D, RecorderSet2D
from eigensimpy.ussim.Boundaries import SimplePML, PMLSettings
from eigensimpy.ussim.Transducers import ReceiverSet2D, Receiver, LinearArray
from eigensimpy.dsp.Signals import Dimension, Quantity
from eigensimpy.dsp.SignalFactories import PulseFactory

import numpy as np

# Create a pulse
pf = PulseFactory()
pf.sampling_freq = 1/6e-10

signal = pf.calc_pulse('simple2')
rcv_signal = signal.copy()
rcv_signal.dims[0] = rcv_signal.dims[0].new(delta=rcv_signal.dims[0].delta*1e-6, 
                                            offset=rcv_signal.dims[0].offset*1e-6 ) 

signal *= 0.4e-2
signal = signal.crop(0.11, 0.51)

signal.dims[0] = signal.dims[0].new(delta=signal.dims[0].delta*1e-6, 
                                    offset=signal.dims[0].offset*1e-6,
                                    si_unit="s") 

# Define the media
dt = signal.dims[0].delta

dx = 0.02 * 1e3
dx *= 1e-6

Nx = 250
Ny = 250


dim_x = Dimension(quantity=Quantity(name="X", si_unit="m"), delta=dx)
dim_y = Dimension(quantity=Quantity(name="Y", si_unit="m"), delta=dx)


sos = 2820
material = ViscousMaterial( comp_sos=sos, shear_sos=1500, density=1732, comp_atten=1.5, shear_atten=5.5)
media = material.to_media( [Ny, Nx], [dim_y, dim_x])

media = media.convert_to_elastic()

# Define a linear transducer
lin_array = LinearArray(
                        unit='m',
                        position=np.array([1e3, 0.5e3, 1]) * 1e-6,
                        pitch=1.5e3 * 1e-6,
                        kerf=0.1 * 1e-6,
                        number_elements=4,
                        element_width=1e3 * 1e-6,
                        emitted_signal=signal,
                        use_shear_wave=False
                    )

emitter_set = lin_array.create_emitters([dim_y, dim_x])

# Define receivers
rdx = 10
receiver_pos_x = np.arange(50, 151, rdx)
receiver_pos_y = np.zeros_like(receiver_pos_x) + 150
receiver_pos_z = np.ones_like(receiver_pos_x)


rcv_signal.data[:] = 0
noffset = rcv_signal.dims[0].offset + rdx * dx / sos;
rcv_signal.dims[0] = rcv_signal.dims[0].new( offset = noffset );

pos = [receiver_pos_x, receiver_pos_y, receiver_pos_z];
receiver_vel1 = Receiver(position=np.array(pos), signal=rcv_signal.copy())
receiver_vel2 = Receiver(position=np.array(pos), signal=rcv_signal.copy())

receiver_set = ReceiverSet2D(   receiver_vel1=receiver_vel1, 
                                receiver_vel2=receiver_vel2)

# Define settings and recorders
settings = SimSettings()
settings.duration = 1 * 1e-3
settings.duration_unit = signal.dims[0].si_unit

recorder1 = FieldDisplay2D(name="Vel1")
recorder2 = FieldDisplay2D(name="Vel2")

recorder1.clim = [-10e-9, 10e-9]
recorder2.clim = recorder1.clim

recorder_set = RecorderSet2D( recorder_vel1=recorder1, recorder_vel2=recorder2)

# Define PML settings and create PML
pml_setting = PMLSettings(attenuation=5,
                          delta_t=dt,
                          delta_x=dx,
                          speed_of_sound=sos,
                          num_elements=30,
                          );
pml = SimplePML(pml_setting)

# Create and run the simulator
simulator = VirieuxDerivViscous2D( media=media, 
                                   emitter=emitter_set, 
                                   receiver=receiver_set,
                                   settings=settings, 
                                   recorder=recorder_set, 
                                   pml=pml)
simulator.run_simulation()