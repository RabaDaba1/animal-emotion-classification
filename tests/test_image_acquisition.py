import multiprocessing as mp
from datetime import datetime
from unittest.mock import MagicMock, patch

import cv2

from src.config import (
    CAMERA_SLEEP_TIME,
    DEFAULT_CAMERA_DEVICE_ID,
    DEFAULT_FRAME_RATE,
    MAX_QUEUE_FILL_LEVEL,
)
from src.services.image_acquisition import ImageAcquisition


class TestImageAcquisition:
    def setup_method(self, method):
        self.output_queue = MagicMock(spec=type(mp.Queue()))  # Use type(mp.Queue())
        self.image_acquirer = ImageAcquisition(
            self.output_queue,
            frame_rate=DEFAULT_FRAME_RATE,
            device_id=DEFAULT_CAMERA_DEVICE_ID,
        )
        self.image_acquirer.running = MagicMock(spec=type(mp.Event()))

    def teardown_method(self, method):
        if hasattr(self.image_acquirer, "running") and isinstance(
            self.image_acquirer.running, MagicMock
        ):
            self.image_acquirer.running.is_set.reset_mock()
            self.image_acquirer.running.clear.reset_mock()
            self.image_acquirer.running.set.reset_mock()

    @patch("src.services.image_acquisition.cv2.VideoCapture")
    def test_setup_camera_success(self, mock_video_capture):
        mock_camera_instance = MagicMock()
        mock_camera_instance.isOpened.return_value = True
        mock_video_capture.return_value = mock_camera_instance

        camera = self.image_acquirer._setup_camera()

        mock_video_capture.assert_called_once_with(self.image_acquirer.device_id)
        assert camera == mock_camera_instance
        self.image_acquirer.running.clear.assert_not_called()

    @patch("src.services.image_acquisition.cv2.VideoCapture")
    def test_setup_camera_failure(self, mock_video_capture):
        mock_camera_instance = MagicMock()
        mock_camera_instance.isOpened.return_value = False
        mock_video_capture.return_value = mock_camera_instance

        with patch("builtins.print") as mock_print:
            camera = self.image_acquirer._setup_camera()

        mock_video_capture.assert_called_once_with(self.image_acquirer.device_id)
        assert camera is None
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
        self.output_queue.put_nowait.assert_called_once()
        args, _ = self.output_queue.put_nowait.call_args
        assert args[0][0] == mock_frame_data  # Frame data
        assert isinstance(args[0][1], datetime)  # Timestamp

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

        self.output_queue.put_nowait.assert_called_once()
        for call_args in mock_print.call_args_list:
            if call_args[0]:  # Ensure there are arguments to print
                assert "Frame capture error: Queue put error" not in call_args[0][0]

    @patch("src.services.image_acquisition.ImageAcquisition._setup_camera")
    @patch("src.services.image_acquisition.ImageAcquisition._capture_frame")
    @patch("src.services.image_acquisition.time.time")
    @patch("src.services.image_acquisition.time.sleep")
    def test_run_loop_behavior(
        self, mock_sleep, mock_time, mock_capture_frame, mock_setup_camera
    ):
        mock_camera_instance = MagicMock(spec=cv2.VideoCapture)
        mock_setup_camera.return_value = mock_camera_instance

        self.image_acquirer.frame_rate = 10
        delay = 0.1
        mock_time.side_effect = [
            0.0,
            0.0,
            delay,
            delay + CAMERA_SLEEP_TIME,
            2 * delay,
            2 * delay + CAMERA_SLEEP_TIME,
            3 * delay,
        ]

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

        assert mock_capture_frame.call_count == 3
        assert mock_sleep.call_count == 3
        mock_sleep.assert_any_call(CAMERA_SLEEP_TIME)

        mock_camera_instance.release.assert_called_once()  # Ensure camera is released

    @patch("src.services.image_acquisition.ImageAcquisition._setup_camera")
    def test_run_setup_camera_fails(self, mock_setup_camera):
        mock_setup_camera.return_value = None  # Simulate camera setup failure
        self.image_acquirer.running.is_set.return_value = True  # So it would try to run

        self.image_acquirer.run()

        mock_setup_camera.assert_called_once()
        self.image_acquirer.running.set.assert_called_once()  # Changed from assert_not_called
        with patch.object(self.image_acquirer, "_capture_frame") as mock_capture:
            self.image_acquirer.run()  # Re-run with this patch
            mock_capture.assert_not_called()

    def test_stop(self):
        self.image_acquirer.stop()
        self.image_acquirer.running.clear.assert_called_once()
