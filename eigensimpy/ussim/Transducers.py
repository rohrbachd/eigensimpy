

import numpy as np
from eigensimpy.dsp.Signals import Signal
from eigensimpy.simmath.MathUtil import to_vec_coords


class LinearArray:
    """ 
    A collection of emitters that represent a linear array



           |              Transducer           |               ^ eleavation (y)
          /                                     \             /
         /                                       \           /
        |                                         |         /
        [x]  [x]  [x]  [x]  [x]  [x].....  [x]  [x]         ----> lateral (x)
        :     :....:          :..:                         |    
        :        :              :                          |
      Position  Pitch          Kerf                        |        
                                                           v axial (z)
                      _ _  
      ElementHeight   _ _ [x] == Element
                          :.: 
                           :
                      ElementWidth   
    """
    def __init__(self, **kwargs):
        self.element_width = kwargs.get('element_width', 0)
        self.element_height = kwargs.get('element_height', 0)
        self.number_elements = kwargs.get('number_elements', 0)
        self.pitch = kwargs.get('pitch', 0)
        self.kerf = kwargs.get('kerf', 0)
        self.unit = kwargs.get('unit', "mm")

        self.position = kwargs.get('position', [0, 0, 0])
        self.emitted_signal = kwargs.get('emitted_signal', None)
        self.use_shear_wave = kwargs.get('use_shear_wave', False)
        
    @property    
    def Z(self):
        return 0;
    
    @property 
    def X(self):
        return 1;
    
    @property
    def Y(self):
        return 2;
    
    def create_emitters(self, dimension):
        ne = self.number_elements

        emitter_stress11 = []
        emitter_stress22 = []

        for i in range(ne):
            emitter = self._create_emitter_element_2d(i, dimension)
            emitter_stress11.append(emitter)
            emitter_stress22.append(emitter.copy())

        if self.use_shear_wave:
            emitters = EmitterSet2D(emitter_stress12=emitter_stress11)
        else:
            emitters = EmitterSet2D(emitter_stress11=emitter_stress11, 
                                    emitter_stress22=emitter_stress22)

        return emitters

    def _create_emitter_element_2d(self, index, dimension):
        x = self.X
        z = self.Z

        delta_z = dimension[z].delta
        offset_z = dimension[z].offset

        delta_x = dimension[x].delta
        offset_x = dimension[x].offset

        txdcr_position = self.position

        ele_width = self.element_width
        kerf = self.kerf

        elemnt_first_pos_x = txdcr_position[x] + (ele_width + kerf) * index
        elemnt_end_pos_x = elemnt_first_pos_x + ele_width - delta_x

        p1_x = round((elemnt_first_pos_x - offset_x) / delta_x)
        p2_x = round((elemnt_end_pos_x - offset_x) / delta_x)

        p_z = round((txdcr_position[z] - offset_z) / delta_z)

        xx, zz, yy = to_vec_coords(range(p1_x, p2_x + 1), p_z, 0)

        if len(xx) > 0:
            position = np.array( list(zip(zz, xx, yy)) )
            emitter = Emitter(position=position, signal=self.emitted_signal)
        else:
            emitter = None  # Empty Emitter

        return emitter
    
        
class ReceiverSet2D:
    
    def __init__(self, **kwargs):
        
        self.receiver_vel1: Receiver = self._ensure_list(kwargs.pop('receiver_vel1', []))
        self.receiver_vel2: Receiver = self._ensure_list(kwargs.pop('receiver_vel2', []))
        self.receiver_stress11: Receiver = self._ensure_list(kwargs.pop('receiver_stress11', []))
        self.receiver_stress22: Receiver = self._ensure_list(kwargs.pop('receiver_stress22', []))
        self.receiver_stress12: Receiver = self._ensure_list(kwargs.pop('receiver_stress12', []))

        self._transducer_manager = TransducerManager([receiver for receiver_list in [self.receiver_vel1,
                                                                                     self.receiver_vel2,
                                                                                     self.receiver_stress11,
                                                                                     self.receiver_stress22,
                                                                                     self.receiver_stress12]
                                                      for receiver in receiver_list])

    @staticmethod
    def _ensure_list(item):
        if isinstance(item, Receiver):
            return [item]
        return item
    
    def record_stress(self, ti, fieldBuffer):
        self.record_stress11(ti, fieldBuffer.stress11)
        self.record_stress22(ti, fieldBuffer.stress22)
        self.record_stress12(ti, fieldBuffer.stress12)

    def record_velocity(self, ti, fieldBuffer):
        self.record_vel1(ti, fieldBuffer.vel1)
        self.record_vel2(ti, fieldBuffer.vel2)
    
    def draw_on_map(self, map, type):
        receiver = getattr(self, type)
        self._transducer_manager.draw_on_map( receiver, map)
        
    def has_no_receivers(self):
        return self._transducer_manager.has_no_transducers()
    
    @property
    def num_receivers(self):
        return self._transducer_manager.num_transducers
        
    @property
    def receivers(self):
        return self._transducer_manager.transducers
    
    @property
    def time_unit(self):
        return self._transducer_manager.time_unit
        
    @property
    def delta(self):
        return self._transducer_manager.delta
        
    @property
    def offset(self):
        return self._transducer_manager.offset
            
    def record_vel1(self, time: float, field: np.ndarray) -> None:
        self._receive_list( time, self.receiver_vel1, field)
             
    def record_vel2(self, time: float, field: np.ndarray) -> None:
        self._receive_list( time, self.receiver_vel2, field)
            
    def record_stress11(self, time: float, field: np.ndarray) -> None:
        self._receive_list( time, self.receiver_stress11, field)
            
    def record_stress22(self, time: float, field: np.ndarray) -> None:
        self._receive_list( time, self.receiver_stress22, field)
            
    def record_stress12(self, time: float, field: np.ndarray) -> None:
        self._receive_list( time, self.receiver_stress12, field)
            
    @staticmethod
    def _receive_list( time: float, receiver_list, field: np.ndarray) -> np.ndarray:
        for receiver in receiver_list:
            receiver.record_signal(time, field)

                
class EmitterSet2D:
    
    def __init__(self, **kwargs):
        self.emitter_vel1 : Emitter = self._ensure_list( kwargs.pop('emitter_vel1', []) )
        self.emitter_vel2 : Emitter = self._ensure_list(kwargs.pop('emitter_vel2', []) )
        self.emitter_stress11 : Emitter = self._ensure_list(kwargs.pop('emitter_stress11', []) )
        self.emitter_stress22 : Emitter = self._ensure_list(kwargs.pop('emitter_stress22', []) )
        self.emitter_stress12 : Emitter = self._ensure_list(kwargs.pop('emitter_stress12', []) )
        
        self._transducer_manager = TransducerManager([emitter for emitter_list in [self.emitter_vel1, 
                                                                                   self.emitter_vel2, 
                                                                                   self.emitter_stress11, 
                                                                                   self.emitter_stress22, 
                                                                                   self.emitter_stress12]
                                                      for emitter in emitter_list])
        
    @staticmethod
    def _ensure_list(item):
        if isinstance(item, Emitter):
            return [item]
        return item
    
    def emit_stress(self, ti, fieldBuffer):
        fieldBuffer.stress11 = self.emit_stress11(ti, fieldBuffer.stress11)
        fieldBuffer.stress22 = self.emit_stress22(ti, fieldBuffer.stress22)
        fieldBuffer.stress12 = self.emit_stress12(ti, fieldBuffer.stress12)
        return fieldBuffer

    def emit_velocity(self, ti, fieldBuffer):
        fieldBuffer.vel1 = self.emit_vel1(ti, fieldBuffer.vel1)
        fieldBuffer.vel2 = self.emit_vel2(ti, fieldBuffer.vel2)
        return fieldBuffer
    
    def draw_on_map(self, map, type):
        emitter = getattr(self, type)
        self._transducer_manager.draw_on_map( emitter, map);
        
    def has_no_emitters(self):
        return self._transducer_manager.has_no_transducers()
        
    @property
    def emitters(self):
        return self._transducer_manager.transducers
                
    @property
    def time_unit(self):
        return self._transducer_manager.time_unit
    
    @property
    def delta(self):
        return self._transducer_manager.delta
        
    @property
    def offset(self):
        return self._transducer_manager.offset
    
    @property
    def num_emitters(self):
        return self._transducer_manager.num_transducers
                
    def emit_vel1(self, time: float, field: np.ndarray) -> np.ndarray:
        return self._emit_list( time, self.emitter_vel1, field)  

    def emit_vel2(self, time: float, field: np.ndarray) -> np.ndarray:
        return self._emit_list( time, self.emitter_vel2, field)   

    def emit_stress11(self, time: float, field: np.ndarray) -> np.ndarray:
        return self._emit_list( time, self.emitter_stress11, field)    
    
    def emit_stress22(self, time: float, field: np.ndarray) -> np.ndarray:
        return self._emit_list( time, self.emitter_stress22, field)    
        
    def emit_stress12(self, time: float, field: np.ndarray) -> np.ndarray:
        return self._emit_list( time, self.emitter_stress12, field)    
    
    
    @staticmethod
    def _emit_list( time: float, emitter_list, field: np.ndarray) -> np.ndarray:
        for emitter in emitter_list:
            field = emitter.emitt_signal(time, field)
        return field
    
class TransducerManager:
    
    def __init__(self, transducers):
        self.transducers = transducers
        self._validate_transducers()
        
    def _validate_transducers(self):
        transducer = self.transducers
        nDevices = len(transducer)
        
        for i in range(1, nDevices):
            sig = transducer[i].signal
            if sig.dims[0].delta != self.delta or sig.dims[0].offset != self.offset or sig.dims[0].quantity.si_unit != self.time_unit:
                raise ValueError("All signals must match in delta, offset, and si_unit")

    def draw_on_map(self, transducer_list, map):
        if transducer_list:
            for transducer in transducer_list:
                if transducer:
                    pos = transducer.position
                    nPos = pos.shape[0]
                    # iterate over all positions
                    for pi in range(nPos):
                        map[pos[pi, 0], pos[pi, 1]] = 1
    
    def has_no_transducers(self):
        return len(self.transducers) == 0
      
    @property            
    def num_transducers(self):
        return len( self.transducers)

    @property
    def time_unit(self):
        transducer = self.transducers
        if transducer:
            return transducer[0].signal.dims[0].quantity.si_unit
        else:
            return ""
        
    @property
    def delta(self):
        transducer = self.transducers
        if transducer:
            return transducer[0].signal.dims[0].delta
        else:
            return 1
        
    @property
    def offset(self):
        transducer = self.transducers
        if transducer:
            return transducer[0].signal.dims[0].offset
        else:
            return 0
        
           
class Emitter:
    
    def __init__(self, **kwargs):
        position : np.ndarray = kwargs.pop('position', None)
        signal : Signal= kwargs.pop('signal', None)

        if position is None:
            raise ValueError("Please provide a 'position' parameter.")
        if signal is None:
            raise ValueError("Please provide a 'signal' parameter.")
        
        num_signals = signal.num_signals(0)
        if num_signals != 1:
            raise ValueError("Signal must have only one signal in the first dimension")
        
        self.position = position
        self.signal = signal

        self.update_num_pos()
        
        
    def copy(self):
        return Emitter( position=self.position.copy(), signal=self.signal.copy() )
        
            
    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        self._position = value
        self.update_num_pos()

    def update_num_pos(self) -> None:
        self.num_pos = self.position.shape[0]

    def emitt_signal(self, time: float, field: np.ndarray) -> np.ndarray:

        shapeField = field.shape
        nDims = len(shapeField)
        
        for i in range(self.num_pos):
            val = self.signal.value_at(time, scip_out_of_bounds=True)
            pos = tuple(self.position[i, :].astype(int))

            field[pos[0:nDims]] = field[ pos[0:nDims] ] + val

        return field
    
class Receiver:
    """
    The Receiver class represents a receiver that can consist of several elements.
    The recorded signals at each element will be summed.
    
    A Receiver can be initialized using a position and signal argument.
    """
    
    def __init__(self, **kwargs):
        position : np.ndarray = kwargs.pop('position', None)
        signal : Signal= kwargs.pop('signal', None)

        if position is None:
            raise ValueError("Please provide a 'position' parameter.")
        if signal is None:
            raise ValueError("Please provide a 'signal' parameter.")
        
        num_signals = signal.num_signals(0)
        if num_signals != 1:
            raise ValueError("Signal must have only one signal in the first dimension")
        
        self.position = position
        self.signal = signal

        self.update_num_pos()

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        self._position = value
        self.update_num_pos()

    def update_num_pos(self) -> None:
        self.num_pos = self.position.shape[1]

    def record_signal(self, time: float, field: np.ndarray) -> None:

        value = 0;
        shapeField = field.shape
        nDims = len(shapeField)
        
        for i in range(self.num_pos):
            pos = tuple(self.position[:, i].astype(int))
            value = value + field[ pos[0:nDims] ]
            
        self.signal.set_value_at( value, time, scip_out_of_bounds=True)   