import multiprocessing as mp
from unittest.mock import MagicMock, patch

import cv2  # Add this import
import numpy as np
import pytest  # Import pytest for raises
import torch
from pytest import approx  # Import approx for float comparisons

from src.config import PROCESS_QUEUE_TIMEOUT
from src.services.emotion_classification import EmotionClassification


class TestEmotionClassification:
    def setup_method(self, method):
        self.input_queue = MagicMock(spec=type(mp.Queue()))  # Use type(mp.Queue())
        self.output_queue = MagicMock(spec=type(mp.Queue()))  # Use type(mp.Queue())
        self.emotion_classifier = EmotionClassification(
            self.input_queue, self.output_queue
        )
        self.emotion_classifier.running = MagicMock(spec=type(mp.Event()))

    def teardown_method(self, method):
        if hasattr(self.emotion_classifier, "running") and isinstance(
            self.emotion_classifier.running, MagicMock
        ):
            self.emotion_classifier.running.is_set.reset_mock()
            self.emotion_classifier.running.clear.reset_mock()
            self.emotion_classifier.running.set.reset_mock()

    @patch(
        "src.services.emotion_classification.EMOTION_CLASSIFICATION_MODEL_PATH"
    )  # Patching the Path object from config
    @patch("src.services.emotion_classification.torch.load")
    @patch("src.services.emotion_classification.timm.create_model")
    @patch("src.services.emotion_classification.transforms.Compose")
    def test_load_model_success(
        self,
        mock_compose,
        mock_create_model,
        mock_torch_load,
        mock_model_path_obj,  # Renamed from mock_path
    ):
        mock_model_instance = MagicMock()
        mock_create_model.return_value = mock_model_instance
        mock_torch_load.return_value = {"model_state_dict": "dummy_state_dict"}

        mock_model_path_obj.exists.return_value = (
            True  # Configure the patched Path object directly
        )

        mock_transform_instance = MagicMock()
        mock_compose.return_value = mock_transform_instance

        self.emotion_classifier._load_model()

        mock_model_path_obj.exists.assert_called_once()
        mock_torch_load.assert_called_once_with(
            mock_model_path_obj, map_location=self.emotion_classifier.device
        )
        mock_create_model.assert_called_once_with(
            "mobilenetv3_large_100", pretrained=False, num_classes=3
        )
        mock_model_instance.load_state_dict.assert_called_once_with("dummy_state_dict")
        mock_model_instance.to.assert_called_once_with(self.emotion_classifier.device)
        mock_model_instance.eval.assert_called_once()
        assert self.emotion_classifier.model == mock_model_instance
        assert self.emotion_classifier.transform == mock_transform_instance
        mock_compose.assert_called_once()

    @patch("src.services.emotion_classification.EMOTION_CLASSIFICATION_MODEL_PATH")
    def test_load_model_file_not_found(
        self, mock_model_path_obj
    ):  # Renamed from mock_path
        mock_model_path_obj.exists.return_value = False

        with pytest.raises(FileNotFoundError):
            self.emotion_classifier._load_model()

    def test_process_next_crops_empty_queue(self):
        self.input_queue.get.side_effect = mp.queues.Empty  # This should be fine
        self.emotion_classifier._process_next_crops()
        self.input_queue.get.assert_called_once_with(timeout=PROCESS_QUEUE_TIMEOUT)
        self.output_queue.put_nowait.assert_not_called()

    @patch.object(EmotionClassification, "_classify_emotions")
    def test_process_next_crops_success(self, mock_classify_emotions):
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_timestamp = "2023-01-01 12:00:00"
        mock_cropped_images = [
            {"image": MagicMock(), "bbox": (0, 0, 10, 10), "class": 0}
        ]
        mock_emotion_results = [
            {"emotion": "happy", "confidence": 0.9, "bbox": (0, 0, 10, 10), "class": 0}
        ]

        self.input_queue.get.return_value = (
            mock_frame,
            mock_timestamp,
            mock_cropped_images,
        )
        mock_classify_emotions.return_value = mock_emotion_results

        self.emotion_classifier._process_next_crops()

        self.input_queue.get.assert_called_once_with(timeout=PROCESS_QUEUE_TIMEOUT)
        mock_classify_emotions.assert_called_once_with(mock_cropped_images)
        self.output_queue.put_nowait.assert_called_once_with(
            (mock_frame, mock_timestamp, mock_emotion_results)
        )

    @patch.object(EmotionClassification, "_classify_emotions")
    def test_process_next_crops_no_results_from_classify(self, mock_classify_emotions):
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_timestamp = "2023-01-01 12:00:00"
        mock_cropped_images = [
            {"image": MagicMock(), "bbox": (0, 0, 10, 10), "class": 0}
        ]

        self.input_queue.get.return_value = (
            mock_frame,
            mock_timestamp,
            mock_cropped_images,
        )
        mock_classify_emotions.return_value = []  # No emotion results

        self.emotion_classifier._process_next_crops()

        self.input_queue.get.assert_called_once_with(timeout=PROCESS_QUEUE_TIMEOUT)
        mock_classify_emotions.assert_called_once_with(mock_cropped_images)
        self.output_queue.put_nowait.assert_not_called()

    def test_classify_emotions_no_model(self):
        self.emotion_classifier.model = None
        results = self.emotion_classifier._classify_emotions([{"image": MagicMock()}])
        assert results == []

    @patch("src.services.emotion_classification.cv2.cvtColor")
    @patch("src.services.emotion_classification.Image.fromarray")
    def test_classify_emotions_success(self, mock_fromarray, mock_cvtcolor):
        self.emotion_classifier.model = MagicMock()

        mock_input_tensor_for_model = (
            "input_tensor"  # This is what the model will receive
        )
        mock_transformed_image = MagicMock()
        mock_unsqueezed_image = MagicMock()

        self.emotion_classifier.transform = MagicMock(
            return_value=mock_transformed_image
        )
        mock_transformed_image.unsqueeze.return_value = mock_unsqueezed_image
        mock_unsqueezed_image.to.return_value = mock_input_tensor_for_model

        self.emotion_classifier.device = "cpu"  # or mock torch.device

        mock_pil_image = MagicMock()
        mock_fromarray.return_value = mock_pil_image

        mock_output_tensor = torch.tensor([[0.1, 0.8, 0.1]])  # happy
        self.emotion_classifier.model.return_value = mock_output_tensor

        with (
            patch(
                "src.services.emotion_classification.torch.nn.functional.softmax"
            ) as mock_softmax,
            patch("src.services.emotion_classification.torch.argmax") as mock_argmax,
        ):
            mock_softmax.return_value = torch.tensor([0.1, 0.8, 0.1])  # Probabilities
            mock_argmax.return_value = torch.tensor([1])  # Index of 'happy'

            crop_image_np = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            cropped_images_data = [
                {"image": crop_image_np, "bbox": (10, 10, 60, 60), "class": "dog"}
            ]

            results = self.emotion_classifier._classify_emotions(cropped_images_data)

            mock_cvtcolor.assert_called_once_with(crop_image_np, cv2.COLOR_BGR2RGB)
            mock_fromarray.assert_called_once()  # Argument is the result of cvtColor
            self.emotion_classifier.transform.assert_called_once_with(mock_pil_image)
            mock_transformed_image.unsqueeze.assert_called_once_with(0)
            mock_unsqueezed_image.to.assert_called_once_with(
                self.emotion_classifier.device
            )
            self.emotion_classifier.model.assert_called_once_with(
                mock_input_tensor_for_model
            )
            mock_softmax.assert_called_once()
            mock_argmax.assert_called_once()

            assert len(results) == 1
            assert (
                results[0]["emotion"] == "happy"
            )  # class_names = ["angry", "happy", "sad"]
            assert results[0]["confidence"] == approx(0.8, abs=1e-5)
            assert results[0]["bbox"] == (10, 10, 60, 60)
            assert results[0]["class"] == "dog"

    def test_classify_emotions_empty_or_invalid_crop_image(self):
        self.emotion_classifier.model = MagicMock()  # Needs a model to proceed
        self.emotion_classifier.transform = MagicMock()

        cropped_images_none = [{"image": None, "bbox": (0, 0, 0, 0), "class": 0}]
        results_none = self.emotion_classifier._classify_emotions(cropped_images_none)
        assert results_none == []

        crop_image_zerosize = np.array([])
        cropped_images_zerosize = [
            {"image": crop_image_zerosize, "bbox": (0, 0, 0, 0), "class": 0}
        ]
        results_zerosize = self.emotion_classifier._classify_emotions(
            cropped_images_zerosize
        )
        assert results_zerosize == []

        crop_image_2d = np.zeros((50, 50), dtype=np.uint8)
        cropped_images_2d = [
            {"image": crop_image_2d, "bbox": (0, 0, 10, 10), "class": 0}
        ]
        results_2d = self.emotion_classifier._classify_emotions(cropped_images_2d)
        assert results_2d == []

        crop_image_4channel = np.zeros((50, 50, 4), dtype=np.uint8)
        cropped_images_4channel = [
            {"image": crop_image_4channel, "bbox": (0, 0, 10, 10), "class": 0}
        ]
        results_4channel = self.emotion_classifier._classify_emotions(
            cropped_images_4channel
        )
        assert results_4channel == []

    def test_run_method_calls_load_model_and_process_loop(self):
        self.emotion_classifier._load_model = MagicMock()
        self.emotion_classifier._process_next_crops = MagicMock()

        self.emotion_classifier.running.is_set.side_effect = [True, False]

        self.emotion_classifier.run()

        self.emotion_classifier.running.set.assert_called_once()
        self.emotion_classifier._load_model.assert_called_once()
        self.emotion_classifier._process_next_crops.assert_called_once()  # Called once due to side_effect

    def test_run_method_handles_exception_in_loop(self):
        self.emotion_classifier._load_model = MagicMock()
        self.emotion_classifier._process_next_crops = MagicMock(
            side_effect=Exception("Test loop error")
        )
        self.emotion_classifier.running.is_set.side_effect = [True, False]  # Loop once

        with patch("builtins.print") as mock_print:
            self.emotion_classifier.run()

        self.emotion_classifier.running.set.assert_called_once()
        self.emotion_classifier._load_model.assert_called_once()
        self.emotion_classifier._process_next_crops.assert_called_once()
        mock_print.assert_any_call("Error in EmotionClassification: Test loop error")

    def test_stop(self):
        self.emotion_classifier.stop()
        self.emotion_classifier.running.clear.assert_called_once()
