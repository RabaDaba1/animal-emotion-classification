import multiprocessing as mp
from unittest.mock import MagicMock, patch

from src.config import PROCESS_JOIN_TIMEOUT
from src.services.emotion_classification import EmotionClassification
from src.services.image_acquisition import ImageAcquisition
from src.services.image_cropping import ImageCropping  # Corrected: image_cropping
from src.services.object_detection import ObjectDetection
from src.services.process_manager import ProcessManager


class TestProcessManager:
    def setup_method(self, method):
        self.raw_frame_queue = MagicMock(spec=type(mp.Queue()))
        self.detected_objects_queue = MagicMock(spec=type(mp.Queue()))
        self.cropped_pets_queue = MagicMock(spec=type(mp.Queue()))
        self.emotion_results_queue = MagicMock(spec=type(mp.Queue()))

        self.mock_image_acquisition = MagicMock(spec=ImageAcquisition)
        self.mock_object_detection = MagicMock(spec=ObjectDetection)
        self.mock_object_detection.pet_classes = {0: "cat", 1: "dog"}
        self.mock_image_cropping = MagicMock(spec=ImageCropping)
        self.mock_emotion_classification = MagicMock(spec=EmotionClassification)

        self.process_manager = ProcessManager(
            raw_frame_queue=self.raw_frame_queue,
            detected_objects_queue=self.detected_objects_queue,
            cropped_pets_queue=self.cropped_pets_queue,
            emotion_results_queue=self.emotion_results_queue,
            image_acquisition=self.mock_image_acquisition,
            object_detection=self.mock_object_detection,
            image_cropping=self.mock_image_cropping,
            emotion_classification=self.mock_emotion_classification,
        )

    def test_initialization(self):
        assert self.process_manager.raw_frame_queue == self.raw_frame_queue
        assert self.process_manager.image_acquisition == self.mock_image_acquisition
        assert not self.process_manager.running

    def test_start_processes(self):
        self.process_manager.running = False  # Ensure it's not running initially
        with patch("builtins.print") as mock_print:
            self.process_manager.start()

        self.mock_image_acquisition.start.assert_called_once()
        self.mock_object_detection.start.assert_called_once()
        self.mock_image_cropping.start.assert_called_once()
        self.mock_emotion_classification.start.assert_called_once()
        assert self.process_manager.running
        mock_print.assert_any_call("Pipeline started successfully")

    def test_start_already_running(self):
        self.process_manager.running = True  # Simulate already running
        with patch("builtins.print") as mock_print:
            self.process_manager.start()

        self.mock_image_acquisition.start.assert_not_called()
        self.mock_object_detection.start.assert_not_called()
        mock_print.assert_called_with("Pipeline is already running")
        assert self.process_manager.running  # Should remain true

    def test_stop_processes(self):
        self.process_manager.running = True  # Simulate running

        self.process_manager._clear_queues = MagicMock()

        with patch("builtins.print") as mock_print:
            self.process_manager.stop()

        self.mock_image_acquisition.stop.assert_called_once()
        self.mock_object_detection.stop.assert_called_once()
        self.mock_image_cropping.stop.assert_called_once()
        self.mock_emotion_classification.stop.assert_called_once()

        self.mock_image_acquisition.join.assert_called_once_with(
            timeout=PROCESS_JOIN_TIMEOUT
        )
        self.mock_object_detection.join.assert_called_once_with(
            timeout=PROCESS_JOIN_TIMEOUT
        )

        self.process_manager._clear_queues.assert_called_once()
        assert not self.process_manager.running
        mock_print.assert_any_call("Pipeline stopped")

    def test_stop_not_running(self):
        self.process_manager.running = False  # Simulate not running
        self.process_manager._clear_queues = MagicMock()  # To check it's not called

        with patch("builtins.print") as mock_print:
            self.process_manager.stop()

        self.mock_image_acquisition.stop.assert_not_called()
        self.mock_image_acquisition.join.assert_not_called()
        self.process_manager._clear_queues.assert_not_called()
        mock_print.assert_called_with("Pipeline is not running")
        assert not self.process_manager.running  # Should remain false

    def test_clear_queue(self):
        mock_queue = MagicMock(spec=type(mp.Queue()))
        mock_queue.empty.side_effect = [False, False, True]
        mock_queue.get_nowait.side_effect = [
            "item1",
            "item2",
            mp.queues.Empty,
        ]  # mp.queues.Empty is from unittest.mock or multiprocessing? It's from multiprocessing.queues

        self.process_manager._clear_queue(mock_queue)

        assert mock_queue.get_nowait.call_count == 2  # Called twice before empty

    def test_clear_queue_exception_on_get(self):
        mock_queue = MagicMock(spec=type(mp.Queue()))
        mock_queue.empty.side_effect = [False, True]  # One item, then empty
        mock_queue.get_nowait.side_effect = Exception("Queue error")

        try:
            self.process_manager._clear_queue(mock_queue)
        except Exception:
            assert False, (
                "_clear_queue raised an exception unexpectedly"
            )  # Pytest way to fail

        mock_queue.get_nowait.assert_called_once()

    def test_is_running(self):
        self.process_manager.running = True
        assert self.process_manager.is_running()
        self.process_manager.running = False
        assert not self.process_manager.is_running()

    @patch("src.services.process_manager.vis.view_raw_frames")
    def test_view_raw_frames(self, mock_view_raw_frames):
        self.process_manager.view_raw_frames()
        mock_view_raw_frames.assert_called_once_with(
            self.process_manager.is_running, self.raw_frame_queue
        )

    @patch("src.services.process_manager.vis.view_detection_results")
    def test_view_detection_results(self, mock_view_detection_results):
        self.process_manager.view_detection_results()
        mock_view_detection_results.assert_called_once_with(
            self.process_manager.is_running,
            self.detected_objects_queue,
            self.mock_object_detection.pet_classes,
        )

    @patch("src.services.process_manager.vis.view_cropped_pets")
    def test_view_cropped_pets(self, mock_view_cropped_pets):
        self.process_manager.view_cropped_pets()
        mock_view_cropped_pets.assert_called_once_with(
            self.process_manager.is_running, self.cropped_pets_queue
        )

    @patch("src.services.process_manager.vis.view_emotion_results")
    def test_view_emotion_results(self, mock_view_emotion_results):
        self.process_manager.view_emotion_results()
        mock_view_emotion_results.assert_called_once_with(
            self.process_manager.is_running, self.emotion_results_queue
        )

    @patch("src.services.process_manager.vis.view_all_outputs")
    def test_view_all_outputs(self, mock_view_all_outputs):
        self.process_manager.view_all_outputs()
        expected_queues_tuple = (
            self.raw_frame_queue,
            self.detected_objects_queue,
            self.cropped_pets_queue,
            self.emotion_results_queue,
        )
        mock_view_all_outputs.assert_called_once_with(
            self.process_manager.is_running,
            expected_queues_tuple,
            self.mock_object_detection.pet_classes,
        )
