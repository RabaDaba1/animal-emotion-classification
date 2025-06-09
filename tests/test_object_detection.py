import multiprocessing as mp
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from pytest import approx  # Import approx for float comparisons

from src.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODELS_DIR,  # Add this import
    PET_CLASSES,
    PROCESS_QUEUE_TIMEOUT,
    YOLO_MODEL_NAME,
)
from src.services.object_detection import ObjectDetection


class TestObjectDetection:
    def setup_method(self, method):
        self.input_queue = MagicMock(
            spec=type(mp.Queue())
        )  # Use type(mp.Queue()) for spec
        self.output_queue = MagicMock(
            spec=type(mp.Queue())
        )  # Use type(mp.Queue()) for spec

        with patch("src.services.object_detection.YOLO") as self.mock_yolo_constructor:
            self.mock_yolo_model_instance = MagicMock()
            self.mock_yolo_constructor.return_value = self.mock_yolo_model_instance
            self.object_detector = ObjectDetection(
                self.input_queue,
                self.output_queue,
                confidence=DEFAULT_CONFIDENCE_THRESHOLD,
            )
        self.object_detector.running = MagicMock(spec=type(mp.Event()))

    def teardown_method(self, method):
        if hasattr(self.object_detector, "running") and isinstance(
            self.object_detector.running, MagicMock
        ):
            self.object_detector.running.is_set.reset_mock()
            self.object_detector.running.clear.reset_mock()
            self.object_detector.running.set.reset_mock()

    def test_initialization(self):
        self.mock_yolo_constructor.assert_called_once_with(
            MODELS_DIR / YOLO_MODEL_NAME
        )  # Updated assertion
        assert self.object_detector.input_queue == self.input_queue
        assert self.object_detector.output_queue == self.output_queue
        assert self.object_detector.confidence == DEFAULT_CONFIDENCE_THRESHOLD
        assert self.object_detector.pet_classes == PET_CLASSES
        assert self.object_detector.yolo_model is not None

    def test_process_next_frame_empty_queue(self):
        self.input_queue.get.side_effect = (
            mp.queues.Empty
        )  # This should be fine if input_queue is correctly mocked
        self.object_detector._process_next_frame()
        self.input_queue.get.assert_called_once_with(timeout=PROCESS_QUEUE_TIMEOUT)
        self.output_queue.put_nowait.assert_not_called()

    @patch.object(ObjectDetection, "_detect_pets")
    def test_process_next_frame_success(self, mock_detect_pets):
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_timestamp = "2023-01-01 12:00:00"
        mock_detections = [{"bbox": (0, 0, 10, 10), "confidence": 0.9, "class": 0}]

        self.input_queue.get.return_value = (mock_frame, mock_timestamp)
        mock_detect_pets.return_value = mock_detections

        self.object_detector._process_next_frame()

        self.input_queue.get.assert_called_once_with(timeout=PROCESS_QUEUE_TIMEOUT)
        mock_detect_pets.assert_called_once_with(mock_frame)
        self.output_queue.put_nowait.assert_called_once_with(
            (mock_frame, mock_timestamp, mock_detections)
        )

    def test_detect_pets_success(self):
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_box1 = MagicMock()
        mock_box1.cls = torch.tensor([0.0])  # Cat
        mock_box1.xyxy = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        mock_box1.conf = torch.tensor([0.95])

        mock_box2 = MagicMock()  # Non-pet class, e.g. 5
        mock_box2.cls = torch.tensor([5.0])
        mock_box2.xyxy = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        mock_box2.conf = torch.tensor([0.8])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box1, mock_box2]

        self.object_detector.yolo_model.return_value = [mock_result]
        self.object_detector.pet_classes = {
            "cat_label": 0,
            "dog_label": 1,
        }  # Example, actual values from src.config

        detections = self.object_detector._detect_pets(mock_frame)

        self.object_detector.yolo_model.assert_called_once_with(
            mock_frame, conf=self.object_detector.confidence
        )
        assert len(detections) == 1
        assert detections[0]["bbox"] == (10, 10, 50, 50)
        assert detections[0]["confidence"] == approx(0.95)
        assert detections[0]["class"] == 0  # Class ID for cat

    def test_detect_pets_no_relevant_detections(self):
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_box_non_pet = MagicMock()
        mock_box_non_pet.cls = torch.tensor([99.0])  # A class not in PET_CLASSES
        mock_box_non_pet.xyxy = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        mock_box_non_pet.conf = torch.tensor([0.9])

        mock_result = MagicMock()
        mock_result.boxes = [mock_box_non_pet]
        self.object_detector.yolo_model.return_value = [mock_result]
        self.object_detector.pet_classes = {"cat": 0, "dog": 1}

        detections = self.object_detector._detect_pets(mock_frame)
        assert len(detections) == 0

    def test_detect_pets_yolo_exception(self):
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.object_detector.yolo_model.side_effect = Exception("YOLO error")

        with patch("builtins.print") as mock_print:
            detections = self.object_detector._detect_pets(mock_frame)

        assert detections == []
        mock_print.assert_any_call("Detection error: YOLO error")

    def test_run_method_calls_process_loop(self):
        self.object_detector._process_next_frame = MagicMock()
        self.object_detector.running.is_set.side_effect = [True, False]  # Loop once

        self.object_detector.run()

        self.object_detector.running.set.assert_called_once()
        self.object_detector._process_next_frame.assert_called_once()

    def test_run_method_handles_exception_in_loop(self):
        self.object_detector._process_next_frame = MagicMock(
            side_effect=Exception("Test loop error")
        )
        self.object_detector.running.is_set.side_effect = [True, False]  # Loop once

        with patch("builtins.print") as mock_print:
            self.object_detector.run()

        self.object_detector.running.set.assert_called_once()
        self.object_detector._process_next_frame.assert_called_once()
        mock_print.assert_any_call("Error in ObjectDetection: Test loop error")

    def test_stop(self):
        self.object_detector.stop()
        self.object_detector.running.clear.assert_called_once()
