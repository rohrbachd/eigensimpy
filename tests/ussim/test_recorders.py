import unittest

# Import square function
from eigensimpy.ussim.Recorders import FieldDisplay2D, MessageFieldRecorder, RecorderList, RecorderSet2D
import numpy as np

class TestFieldDisplay2D(unittest.TestCase):

    def test_field_display_2d_initialize(self):
        field = np.random.rand(5, 5)
        recorder = FieldDisplay2D()

        # Test first initialization
        success = recorder.initialize(field)
        self.assertTrue(success)
        self.assertIsNotNone(recorder.fig)
        self.assertIsNotNone(recorder.axes)
        self.assertIsNotNone(recorder.img)

        # Store initial state
        initial_fig = recorder.fig
        initial_axes = recorder.axes
        initial_img = recorder.img

        # Test second initialization (reusing existing fig, axes, and img)
        success = recorder.initialize(field)
        self.assertTrue(success)
        self.assertIs(recorder.fig, initial_fig)
        self.assertIs(recorder.axes, initial_axes)
        self.assertIsNot(recorder.img, initial_img)  # New img is created
        
    def test_field_display_2d_record(self):
        field = np.random.rand(5, 5)
        recorder = FieldDisplay2D()

        success = recorder.initialize(field)
        self.assertTrue(success)

        success = recorder.record(field)
        displayed_data = recorder.get_displayed_data()
        self.assertTrue(success)
        self.assertTrue( np.allclose(field, displayed_data) )


class TestMessageFieldRecorder(unittest.TestCase):

    def test_message_field_recorder(self):
        field = np.random.rand(5, 5)
        recorder = MessageFieldRecorder()

        success = recorder.initialize(field)
        self.assertTrue(success)
        self.assertEqual(len(recorder.get_messages()), 1)

        success = recorder.record(field)
        self.assertTrue(success)
        self.assertEqual(len(recorder.get_messages()), 2)


class TestRecorderList(unittest.TestCase):

    def test_recorder_set(self):
        field = np.random.rand(5, 5)
        recorder1 = FieldDisplay2D()
        recorder2 = MessageFieldRecorder()

        recorders = RecorderList([recorder1, recorder2])

        self.assertEqual(len(recorders), 2)

        success = recorders.initialize(field)
        self.assertTrue(all(success))

        success = recorders.record(field)
        self.assertTrue(all(success))

        self.assertEqual(len(recorder2.get_messages()), 2)
        
class TestRecorderSet2D(unittest.TestCase):

    def test_recorder_set2d(self):
        field = np.random.rand(5, 5)
        recorder1 = FieldDisplay2D()
        recorder2 = MessageFieldRecorder()
        recorder3 = FieldDisplay2D()
        recorder4 = MessageFieldRecorder()

        recorder_set = RecorderSet2D(recorder_vel1=recorder1,
                                      recorder_vel2=[recorder2, recorder3],
                                      recorder_stress11=recorder4)

        # Check if the lists were created correctly
        self.assertIsInstance(recorder_set.recorder_vel1, RecorderList)
        self.assertIsInstance(recorder_set.recorder_vel2, RecorderList)
        self.assertIsInstance(recorder_set.recorder_stress11, RecorderList)
        self.assertIsInstance(recorder_set.recorder_stress22, RecorderList)
        self.assertIsInstance(recorder_set.recorder_stress12, RecorderList)

        # Test if recorders were added correctly
        self.assertEqual(len(recorder_set.recorder_vel1), 1)
        self.assertEqual(len(recorder_set.recorder_vel2), 2)
        self.assertEqual(len(recorder_set.recorder_stress11), 1)
        self.assertEqual(len(recorder_set.recorder_stress22), 0)
        self.assertEqual(len(recorder_set.recorder_stress12), 0)

        # recorder1.initialize(field)
        # recorder2.initialize(field)
        # recorder3.initialize(field)
        # recorder4.initialize(field)
        recorder_set.initialize(field);
        # Test recording
        recorder_set.record_vel1(field)
        recorder_set.record_vel2(field)
        recorder_set.record_stress11(field)
        recorder_set.record_stress11(field)

        # Test if the MessageFieldRecorder recorded the field
        self.assertEqual(len(recorder2.get_messages()), 2)
        self.assertEqual(len(recorder4.get_messages()), 3)
        
if __name__ == "__main__":
    unittest.main()