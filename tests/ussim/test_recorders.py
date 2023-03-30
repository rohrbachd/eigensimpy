import unittest

# Import square function
from eigensimpy.ussim.Recorders import FieldDisplay2D, MessageFieldRecorder, RecorderSet
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


class TestRecorderSet(unittest.TestCase):

    def test_recorder_set(self):
        field = np.random.rand(5, 5)
        recorder1 = FieldDisplay2D()
        recorder2 = MessageFieldRecorder()

        recorders = RecorderSet([recorder1, recorder2])

        self.assertEqual(len(recorders), 2)

        success = recorders.initialize(field)
        self.assertTrue(all(success))

        success = recorders.record(field)
        self.assertTrue(all(success))

        self.assertEqual(len(recorder2.get_messages()), 2)
        

if __name__ == "__main__":
    unittest.main()