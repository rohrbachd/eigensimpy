import unittest

# Import square function
from eigensimpy.ussim.Transducers import Emitter, Receiver, EmitterSet2D, ReceiverSet2D, LinearArray
from eigensimpy.dsp.Signals import Signal, Dimension, DimensionArray, Quantity
import numpy as np


class TestLinearArray(unittest.TestCase):

    def test_create_emitters(self):
        # Create a signal for the LinearArray
        dims = [Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.1, offset=1.4)]
        data = np.ones(10)
        sig = Signal(data=data, dims=dims)

        #   0  .1  .2  .3  .4  .5  .6  .7  .8  .9  1.0 1.1 1.2 1.3 1.4 1.5 1.6 
        # [**][**][--][**][**][  ][**][**][--][**][**][  ][**][**][--][**][**]    
        # |      Em1         |    |      Em2         |    |      Em3         |
        #         :                        :         :    :
        #         :------------------------:         :----:
        #                    Pitch                     kerf
        # Create a LinearArray
        linear_array = LinearArray(element_width=0.5,
                                   element_height=0.5,
                                   number_elements=3,
                                   pitch=0.6,
                                   kerf=0.1,
                                   unit="mm",
                                   position=[0, 0, 0],
                                   emitted_signal=sig,
                                   use_shear_wave=False)

        # Create dimensions for create_emitters function
        dim_z = Dimension(quantity=Quantity(name="depth", si_unit="mm"), delta=0.1, offset=0)
        dim_x = Dimension(quantity=Quantity(name="width", si_unit="mm"), delta=0.1, offset=0)
        dimension = [dim_z, dim_x]

        # Call create_emitters function
        emitters = linear_array.create_emitters(dimension)

        # Validate the result
        self.assertEquals(emitters.num_emitters, 6)  # 3 for emitter_stress11 and 3 for emitter_stress22
        self.assertEquals(len(emitters.emitters), 6)

        expected_position0 = np.array([ [0, 0, 0],
                                        [0, 1, 0],
                                        [0, 2, 0],
                                        [0, 3, 0],
                                        [0, 4, 0]])
        
        expected_position1 = np.array([ [0, 6, 0],
                                        [0, 7, 0],
                                        [0, 8, 0],
                                        [0, 9, 0],
                                        [0, 10, 0]])
        
        expected_position2 = np.array([ [0, 12, 0],
                                        [0, 13, 0],
                                        [0, 14, 0],
                                        [0, 15, 0],
                                        [0, 16, 0]])
        
        expected_position =  [expected_position0,
                              expected_position1,
                              expected_position2]
        
        # Validate position of the emitters
        for i in range(3):
            emitter_stress11 = emitters.emitters[i]
            emitter_stress22 = emitters.emitters[i + 3]

            # Check if the position of the emitter_stress11 and emitter_stress22 are the same
            np.testing.assert_array_equal(emitter_stress11.position, emitter_stress22.position)
            np.testing.assert_array_equal(emitters.emitter_stress11[i].position, emitter_stress11.position)

            np.testing.assert_array_almost_equal(emitter_stress11.position, expected_position[i], decimal=1)
            np.testing.assert_array_almost_equal(emitter_stress22.position, expected_position[i], decimal=1)

        # Validate the emitted_signal of the emitters
        for emitter in emitters.emitters:
            self.assertEqual(emitter.signal.dims, sig.dims)
            self.assertEqual(emitter.signal.amplitude, sig.amplitude)
            np.testing.assert_array_almost_equal(emitter.signal.data, sig.data)

class TestReceiverSet2D(unittest.TestCase):
    
    def test_receiver_validation_pass(self):

        # Create three receivers with the same signal dimensions
        dims = [Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.1, offset=1.4)]
        data = np.ones(10)
        sig = Signal(data=data, dims=dims)

        # all have the same signal 
        r1 = Receiver(position=np.array([[0, 0], [0, 1]]), signal=sig)
        r2 = Receiver(position=np.array([[1, 0], [1, 1]]), signal=sig)
        r3 = Receiver(position=np.array([[2, 0], [2, 1]]), signal=sig)

        # Create receiver set with three receivers
        rs = ReceiverSet2D(receiver_vel1=r1, receiver_vel2=r2, receiver_stress11=r3)

        # Validate receivers, should pass
        self.assertEquals(rs.num_receivers, 3)
        self.assertEquals(len(rs.receivers), 3)
        self.assertEquals(rs.delta, 0.1)
        self.assertEquals(rs.offset, 1.4)
        self.assertEquals(rs.time_unit, "s")
        
    def test_receiver_validation_fail(self):
        
        # Different delta
        # Create three receivers with the same signal dimensions
        dims1 = Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.1, offset=0)
        dims2 = Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.2, offset=0)
        
        data = np.ones(10)
        sig1 = Signal(data=data, dims=dims1)
        sig2 = Signal(data=data, dims=dims2)
        sig3 = Signal(data=data, dims=dims1)
        
        r1 = Receiver(position=np.array([[0, 0], [0, 1]]), signal=sig1)
        r2 = Receiver(position=np.array([[1, 0], [1, 1]]), signal=sig2)
        r3 = Receiver(position=np.array([[2, 0], [2, 1]]), signal=sig3)
        
        with self.assertRaises(ValueError):
            ReceiverSet2D(receiver_vel1=r1, receiver_vel2=r2, receiver_stress11=r3)
            
        sig2.dims[0] = dims1.new(offset=1)
        
        r1 = Receiver(position=np.array([[0, 0], [0, 1]]), signal=sig1)
        r2 = Receiver(position=np.array([[1, 0], [1, 1]]), signal=sig2)
        
        with self.assertRaises(ValueError):
            ReceiverSet2D(receiver_vel2=r2, receiver_stress11=r1)
            
        sig2.dims[0] = dims1
        sig3.dims[0] = dims1.new(si_unit="MHz", name="frequency")
        
        r1 = Receiver(position=np.array([[0, 0], [0, 1]]), signal=sig1)
        r2 = Receiver(position=np.array([[1, 0], [1, 1]]), signal=sig2)
        r3 = Receiver(position=np.array([[2, 0], [2, 1]]), signal=sig3)
        
        with self.assertRaises(ValueError):
            ReceiverSet2D(receiver_vel2=r2, receiver_stress11=r1, receiver_stress12=r3)
        
        # this should pass now    
        sig3.dims[0] = dims1
        r1 = Receiver(position=np.array([[0, 0], [0, 1]]), signal=sig1)
        r2 = Receiver(position=np.array([[1, 0], [1, 1]]), signal=sig2)
        r3 = Receiver(position=np.array([[2, 0], [2, 1]]), signal=sig3)
        
        rs = ReceiverSet2D(receiver_vel2=r2, receiver_stress11=r1, receiver_stress12=r3)
        
        # Validate receivers, should pass
        self.assertEquals(rs.num_receivers, 3)
        self.assertEquals(len(rs.receivers), 3)
        self.assertEquals(rs.delta, 0.1)
        self.assertEquals(rs.offset, 0)
        self.assertEquals(rs.time_unit, "s")
    
class TestEmitterSet2D(unittest.TestCase):

    def test_emitter_validation_pass(self):
        
        # Create three emitters with the same signal dimensions
        dims = [Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.1, offset=1.4)]
        data = np.ones(10)
        sig = Signal(data=data, dims=dims)
        
        # all have the same signal 
        e1 = Emitter(position=np.array([[0, 0], [0, 1]]), signal=sig)
        e2 = Emitter(position=np.array([[1, 0], [1, 1]]), signal=sig)
        e3 = Emitter(position=np.array([[2, 0], [2, 1]]), signal=sig)
        
        # Create emitter set with three emitters
        es = EmitterSet2D(emitter_vel1=e1, emitter_vel2=e2, emitter_stress11=e3)
        
        # Validate emitters, should pass
        self.assertEquals( es.num_emitters, 3);
        self.assertEquals( len(es.emitters), 3);
        self.assertEquals( es.delta, 0.1);
        self.assertEquals( es.offset, 1.4);
        self.assertEquals( es.time_unit, "s");
        
        
    def test_emitter_validation_fail(self):
        
        # Different delta
        # Create three emitters with the same signal dimensions
        dims1 = Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.1, offset=0)
        dims2 = Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.2, offset=0)
        
        data = np.ones(10)
        sig1 = Signal(data=data, dims=dims1)
        sig2 = Signal(data=data, dims=dims2)
        sig3 = Signal(data=data, dims=dims1)
        
        e1 = Emitter(position=np.array([[0, 0], [0, 1]]), signal=sig1)
        e2 = Emitter(position=np.array([[1, 0], [1, 1]]), signal=sig2)
        e3 = Emitter(position=np.array([[2, 0], [2, 1]]), signal=sig3)
        
        with self.assertRaises(ValueError):
            EmitterSet2D(emitter_vel1=e1, emitter_vel2=e2, emitter_stress11=e3)
            
        sig2.dims[0] = dims1.new( offset=1)
        
        e1 = Emitter(position=np.array([[0, 0], [0, 1]]), signal=sig1)
        e2 = Emitter(position=np.array([[1, 0], [1, 1]]), signal=sig2)
        
        with self.assertRaises(ValueError):
            EmitterSet2D( emitter_vel2=e2, emitter_stress11=e1)
            
        sig2.dims[0] = dims1
        sig3.dims[0] = dims1.new( si_unit="MHz", name="frequency" )
        
        e1 = Emitter(position=np.array([[0, 0], [0, 1]]), signal=sig1)
        e2 = Emitter(position=np.array([[1, 0], [1, 1]]), signal=sig2)
        e3 = Emitter(position=np.array([[2, 0], [2, 1]]), signal=sig3)
        
        with self.assertRaises(ValueError):
            EmitterSet2D( emitter_vel2=e2, emitter_stress11=e1, emitter_stress12=e3)
         
        # this should pass now    
        sig3.dims[0] = dims1
        e1 = Emitter(position=np.array([[0, 0], [0, 1]]), signal=sig1)
        e2 = Emitter(position=np.array([[1, 0], [1, 1]]), signal=sig2)
        e3 = Emitter(position=np.array([[2, 0], [2, 1]]), signal=sig3)
        
        es = EmitterSet2D( emitter_vel2=e2, emitter_stress11=e1, emitter_stress12=e3)
        
        # Validate emitters, should pass
        self.assertEquals( es.num_emitters, 3);
        self.assertEquals( len(es.emitters), 3);
        self.assertEquals( es.delta, 0.1);
        self.assertEquals( es.offset, 0);
        self.assertEquals( es.time_unit, "s");
            
            
            
    def test_draw_on_map(self):
        
        map = np.zeros((10, 10))
        
        data = np.array([1, 2, 3, 4, 5, 6, 7])
        dimTime = Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2);
        dims = DimensionArray([dimTime])
        signal = Signal(data=data, dims=dims)
        
        # Set up an emitter at position 3
        position = np.array( [ [ 0, 3 ], 
                               [ 1, 0 ], 
                               [ 4, 5 ] ] , dtype=np.uint64)
        emitter = Emitter(position=position, signal=signal)
         
        emitter_set = EmitterSet2D(
            emitter_vel1 = emitter
        )
        
        emitter_set.draw_on_map(map, 'emitter_vel2')
        np.testing.assert_equal(map, np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ))
        
        emitter_set.draw_on_map(map, 'emitter_vel1')
        np.testing.assert_equal(map, np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ))
        
    def test_emitter_validation_array(self):
        
        # Create three emitters with the same signal dimensions
        dims = [Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.1, offset=1.4)]
        data = np.ones(10)
        sig = Signal(data=data, dims=dims)
        
        # all have the same signal 
        e1 = Emitter(position=np.array([[0, 0], [0, 1]]), signal=sig)
        e2 = Emitter(position=np.array([[1, 0], [1, 1]]), signal=sig)
        e3 = Emitter(position=np.array([[2, 0], [2, 1]]), signal=sig)
        e4 = Emitter(position=np.array([[3, 0], [3, 1]]), signal=sig)
        
        # Create emitter set with arrays of emitters
        es = EmitterSet2D(emitter_vel1=[e1, e2], emitter_vel2=[e3, e4])
        
        # Validate emitters, should pass
        self.assertEquals( es.num_emitters, 4);
        self.assertEquals( len(es.emitters), 4);
        self.assertEquals( es.delta, 0.1);
        self.assertEquals( es.offset, 1.4);
        self.assertEquals( es.time_unit, "s");
                
class TestEmitter(unittest.TestCase):

        
    def test_emitt_signal(self):
        
        data = np.array([1, 2, 3, 4, 5, 6, 7])
        dimTime = Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2);
        dims = DimensionArray([dimTime])
        signal = Signal(data=data, dims=dims)
        
        # Set up an emitter at position 3
        position = np.array( [ [ 0, 3 ], 
                               [ 1, 0 ], 
                               [ 4, 5 ] ] , dtype=np.uint64)
        emitter = Emitter(position=position, signal=signal)

        # Set up a medium with the same shape as the signal
        field = np.zeros( [10, 20] )
        expected_field = field.copy();
        
        # Emit the signal at time index 2 = sample 0 
        field = emitter.emitt_signal(2, field)
        expected_field[ 0, 3] = 1;
        expected_field[ 1, 0] = 1;
        expected_field[ 4, 5] = 1;
        
        np.testing.assert_array_equal(field, expected_field)
        
        field = emitter.emitt_signal( 3.5, field)
        expected_field[ 0, 3] = 1+4;
        expected_field[ 1, 0] = 1+4;
        expected_field[ 4, 5] = 1+4;
        np.testing.assert_array_equal(field, expected_field)
        
        # Test that an error is raised if the signal has zero signals in the first dimension
        data = np.array( np.array( [1,2,3,4,5,6,7,8] ).reshape( 8, 1) )
        dims = DimensionArray([ dimTime,
                                Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=2.0, offset=3.1)]) # second dimensions
        signal = Signal(data=data, dims=dims)    
        emitter = Emitter(position=position, signal=signal);
        
        field = emitter.emitt_signal( 5.5, field)
        expected_field[ 0, 3] = 5+8;
        expected_field[ 1, 0] = 5+8;
        expected_field[ 4, 5] = 5+8;
        np.testing.assert_array_equal(field, expected_field)
        
    def test_init(self):
        # Test that an error is raised if the signal has more than one dimension
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2),
                                   Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=2.0, offset=3.1)])
        signal = Signal(data=data, dims=dims)
        # Set up an emitter at position 3
        position = np.array( [ [ 0, 3 ], 
                               [ 1, 0 ], 
                               [ 4, 5] ] , dtype=np.uint64)
        
        with self.assertRaises(ValueError):
            Emitter(position=position, signal=signal)

        # Test that an error is raised if the signal has zero signals in the first dimension
        data = np.array( np.array( [1,2,3,4,5,6,7,8] ).reshape( 1, 8) )
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2),
                               Dimension(quantity=Quantity(name="channel", si_unit="V"), delta=2.0, offset=3.1)])
        signal = Signal(data=data, dims=dims)

        with self.assertRaises(ValueError):
            Emitter(position=position, signal=signal)
   
class TestReceiver(unittest.TestCase):         
            
    def test_record_signal(self):
    # Set up a receiver at position 3
    
        position = np.array([[0, 3], [1, 0], [4, 5]], dtype=np.uint64)
        dims = DimensionArray([Dimension(quantity=Quantity(name="time", si_unit="s"), delta=0.5, offset=2),
                               Dimension(quantity=Quantity(name="Sample", si_unit="")) ])
        data = np.zeros([8,1]);
        signal = Signal(data=data, dims=dims)
        
        receiver = Receiver(position=position, signal=signal)

        # Set up a medium 
        field = np.zeros([10, 20])
        field[0, 3 ] = 1.0
        field[1, 0 ] = 2.0
        field[4, 5 ] = 3.0
        
        expected_signal_data = np.zeros([8,1]);
        
        # Record the signal at time index 2 = sample 0 
        receiver.record_signal(2, field)
        expected_signal_data[0] = 1+2+3
        np.testing.assert_equal(receiver.signal.data, expected_signal_data)
        
        # Record the signal at time index 3 = sample 2
        receiver.record_signal(3, field)
        
        expected_signal_data[2] = 1+2+3
        np.testing.assert_equal(receiver.signal.data, expected_signal_data)

        # change field values
        field[:, : ] = 1.5
        
        # Record the signal at time index 5.5 = sample 7
        receiver.record_signal(5.5, field)
        expected_signal_data[7] = 1.5 * 3
        np.testing.assert_equal(receiver.signal.data, expected_signal_data)        
        

if __name__ == "__main__":
    unittest.main()