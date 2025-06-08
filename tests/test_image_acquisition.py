import multiprocessing as mp
import multiprocessing.queues as mpq
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import cv2  # OpenCV is a direct dependency

from config import (
    CAMERA_SLEEP_TIME,
    DEFAULT_CAMERA_DEVICE_ID,
    DEFAULT_FRAME_RATE,
    MAX_QUEUE_FILL_LEVEL,
)
from services.image_acquisition import ImageAcquisition


class TestImageAcquisition(unittest.TestCase):
    def setUp(self):
        self.output_queue = MagicMock(spec=mpq.Queue)
        self.image_acquirer = ImageAcquisition(
            self.output_queue,
            frame_rate=DEFAULT_FRAME_RATE,
            device_id=DEFAULT_CAMERA_DEVICE_ID,
        )
        self.image_acquirer.running = MagicMock(spec=mp.Event)

    def tearDown(self):
        if hasattr(self.image_acquirer, "running") and isinstance(
            self.image_acquirer.running, MagicMock
        ):
            self.image_acquirer.running.is_set.reset_mock()
            self.image_acquirer.running.clear.reset_mock()
            self.image_acquirer.running.set.reset_mock()

    @patch("services.image_acquisition.cv2.VideoCapture")
    def test_setup_camera_success(self, mock_video_capture):
        mock_camera_instance = MagicMock()
        mock_camera_instance.isOpened.return_value = True
        mock_video_capture.return_value = mock_camera_instance

        camera = self.image_acquirer._setup_camera()

        mock_video_capture.assert_called_once_with(self.image_acquirer.device_id)
        self.assertEqual(camera, mock_camera_instance)
        self.image_acquirer.running.clear.assert_not_called()

    @patch("services.image_acquisition.cv2.VideoCapture")
    def test_setup_camera_failure(self, mock_video_capture):
        mock_camera_instance = MagicMock()
        mock_camera_instance.isOpened.return_value = False
        mock_video_capture.return_value = mock_camera_instance

        with patch("builtins.print") as mock_print:
            camera = self.image_acquirer._setup_camera()

        mock_video_capture.assert_called_once_with(self.image_acquirer.device_id)
        self.assertIsNone(camera)
        self.image_acquirer.running.clear.assert_called_once()  # Should clear running event
        mock_print.assert_called_with("Error: Could not open camera.")

    def test_capture_frame_success(self):
        mock_camera = MagicMock(spec=cv2.VideoCapture)
        mock_frame_data = "frame_data"
        mock_camera.read.return_value = (True, mock_frame_data)
        self.output_queue.qsize.return_value = (
            MAX_QUEUE_FILL_LEVEL - 1
        )  # Queue not full

        self.image_acquirer._capture_frame(mock_camera)

        mock_camera.read.assert_called_once()
        # datetime.now() is called, so we can't easily assert exact timestamp
        self.output_queue.put_nowait.assert_called_once()
        args, _ = self.output_queue.put_nowait.call_args
        self.assertEqual(args[0][0], mock_frame_data)  # Frame data
        self.assertIsInstance(args[0][1], datetime)  # Timestamp

    def test_capture_frame_read_failure(self):
        mock_camera = MagicMock(spec=cv2.VideoCapture)
        mock_camera.read.return_value = (False, None)  # Failed to read

        with patch("builtins.print") as mock_print:
            self.image_acquirer._capture_frame(mock_camera)

        mock_camera.read.assert_called_once()
        self.output_queue.put_nowait.assert_not_called()
        mock_print.assert_called_with("Error: Failed to capture image")

    def test_capture_frame_queue_full(self):
        mock_camera = MagicMock(spec=cv2.VideoCapture)
        mock_frame_data = "frame_data"
        mock_camera.read.return_value = (True, mock_frame_data)
        self.output_queue.qsize.return_value = MAX_QUEUE_FILL_LEVEL  # Queue is full

        self.image_acquirer._capture_frame(mock_camera)

        mock_camera.read.assert_called_once()
        self.output_queue.put_nowait.assert_not_called()  # Should not put if queue is full

    def test_capture_frame_put_exception(self):
        mock_camera = MagicMock(spec=cv2.VideoCapture)
        mock_frame_data = "frame_data"
        mock_camera.read.return_value = (True, mock_frame_data)
        self.output_queue.qsize.return_value = 0
        self.output_queue.put_nowait.side_effect = Exception("Queue put error")

        with patch("builtins.print") as mock_print:
            self.image_acquirer._capture_frame(mock_camera)

        # It should still attempt to put
        self.output_queue.put_nowait.assert_called_once()
        # The exception inside _capture_frame is caught and printed by the outer try-except in _capture_frame
        mock_print.assert_any_call("Frame capture error: Queue put error")

    @patch("services.image_acquisition.ImageAcquisition._setup_camera")
    @patch("services.image_acquisition.ImageAcquisition._capture_frame")
    @patch("services.image_acquisition.time.time")
    @patch("services.image_acquisition.time.sleep")
    def test_run_loop_behavior(
        self, mock_sleep, mock_time, mock_capture_frame, mock_setup_camera
    ):
        mock_camera_instance = MagicMock(spec=cv2.VideoCapture)
        mock_setup_camera.return_value = mock_camera_instance

        # Simulate time progression for a few iterations
        # Frame rate 10 -> delay 0.1s
        self.image_acquirer.frame_rate = 10
        delay = 0.1
        # t0, t0+delay, t0+2*delay ...
        mock_time.side_effect = [
            0.0,
            0.0,
            delay,
            delay + CAMERA_SLEEP_TIME,
            2 * delay,
            2 * delay + CAMERA_SLEEP_TIME,
            3 * delay,
        ]

        # Loop 3 times then stop
        self.image_acquirer.running.is_set.side_effect = [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ]

        self.image_acquirer.run()

        mock_setup_camera.assert_called_once()
        # current_time - last_time >= delay
        # Iter 1: 0.0 - 0.0 >= 0.1 (False) -> sleep
        # Iter 2: delay - 0.0 >= 0.1 (True) -> capture, last_time = delay
        # Iter 3: delay+CAMERA_SLEEP_TIME - delay >= 0.1 (True, assuming CAMERA_SLEEP_TIME is small but enough) -> capture, last_time = delay+CAMERA_SLEEP_TIME
        # Iter 4: 2*delay - (delay+CAMERA_SLEEP_TIME) >= 0.1 (depends on CAMERA_SLEEP_TIME)
        # Let's simplify: ensure capture is called, and sleep is called.
        # With the time side_effect:
        # 1. last_time=0.0. current_time=0.0. (0.0-0.0 < 0.1) -> sleep.
        # 2. last_time=0.0. current_time=delay. (delay-0.0 >= 0.1) -> capture. last_time=delay.
        # 3. last_time=delay. current_time=delay+CAMERA_SLEEP_TIME. (delay+CST - delay < 0.1 if CST is small) -> sleep
        # 4. last_time=delay. current_time=2*delay. (2*delay - delay >= 0.1) -> capture. last_time=2*delay
        # 5. last_time=2*delay. current_time=2*delay+CAMERA_SLEEP_TIME. (2*delay+CST - 2*delay < 0.1) -> sleep
        # 6. last_time=2*delay. current_time=3*delay. (3*delay - 2*delay >= 0.1) -> capture. last_time=3*delay

        # Based on the above logic, capture_frame should be called 3 times.
        self.assertEqual(mock_capture_frame.call_count, 3)
        # Sleep should be called when condition is false
        self.assertEqual(mock_sleep.call_count, 3)
        mock_sleep.assert_any_call(CAMERA_SLEEP_TIME)

        mock_camera_instance.release.assert_called_once()  # Ensure camera is released

    @patch("services.image_acquisition.ImageAcquisition._setup_camera")
    def test_run_setup_camera_fails(self, mock_setup_camera):
        mock_setup_camera.return_value = None  # Simulate camera setup failure
        self.image_acquirer.running.is_set.return_value = True  # So it would try to run

        self.image_acquirer.run()

        mock_setup_camera.assert_called_once()
        # Ensure the run method exits early and doesn't enter the loop
        self.image_acquirer.running.set.assert_not_called()  # set is called at start of parent Process.run
        # but here we test the content of our run method
        # The instance's running event is set by the parent mp.Process
        # Our method sets it again, this is fine.
        # The key is that the loop isn't entered.
        # Let's check if _capture_frame was called
        with patch.object(self.image_acquirer, "_capture_frame") as mock_capture:
            self.image_acquirer.run()  # Re-run with this patch
            mock_capture.assert_not_called()

    def test_stop(self):
        self.image_acquirer.stop()
        self.image_acquirer.running.clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
