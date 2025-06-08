import multiprocessing as mp
import multiprocessing.queues as mpq
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from config import PROCESS_QUEUE_TIMEOUT

# Assuming services and config are accessible from the test environment
# Adjust sys.path if necessary, or ensure your test runner handles it.
# For example, by running tests from the project root.
from services.emotion_classification import EmotionClassification


class TestEmotionClassification(unittest.TestCase):
    def setUp(self):
        self.input_queue = MagicMock(spec=mpq.Queue)
        self.output_queue = MagicMock(spec=mpq.Queue)
        self.emotion_classifier = EmotionClassification(
            self.input_queue, self.output_queue
        )
        # Mock the event directly on the instance for easier control in tests
        self.emotion_classifier.running = MagicMock(spec=mp.Event)

    def tearDown(self):
        # Ensure a clean state for running event if it was manipulated
        if hasattr(self.emotion_classifier, "running") and isinstance(
            self.emotion_classifier.running, MagicMock
        ):
            self.emotion_classifier.running.is_set.reset_mock()
            self.emotion_classifier.running.clear.reset_mock()
            self.emotion_classifier.running.set.reset_mock()

    @patch("services.emotion_classification.Path")
    @patch("services.emotion_classification.torch.load")
    @patch("services.emotion_classification.timm.create_model")
    @patch("services.emotion_classification.transforms.Compose")
    def test_load_model_success(
        self, mock_compose, mock_create_model, mock_torch_load, mock_path
    ):
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_create_model.return_value = mock_model_instance
        mock_torch_load.return_value = {"model_state_dict": "dummy_state_dict"}

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value.resolve.return_value = mock_path_instance

        mock_transform_instance = MagicMock()
        mock_compose.return_value = mock_transform_instance

        # Call the method
        self.emotion_classifier._load_model()

        # Assertions
        mock_path.assert_called_once_with("mobilenetv3_pet_emotion_classifier.pth")
        mock_path_instance.exists.assert_called_once()
        mock_torch_load.assert_called_once_with(
            mock_path_instance, map_location=self.emotion_classifier.device
        )
        mock_create_model.assert_called_once_with(
            "mobilenetv3_large_100", pretrained=False, num_classes=3
        )
        mock_model_instance.load_state_dict.assert_called_once_with("dummy_state_dict")
        mock_model_instance.to.assert_called_once_with(self.emotion_classifier.device)
        mock_model_instance.eval.assert_called_once()
        self.assertEqual(self.emotion_classifier.model, mock_model_instance)
        self.assertEqual(self.emotion_classifier.transform, mock_transform_instance)
        mock_compose.assert_called_once()

    @patch("services.emotion_classification.Path")
    def test_load_model_file_not_found(self, mock_path):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value.resolve.return_value = mock_path_instance

        with self.assertRaises(FileNotFoundError):
            self.emotion_classifier._load_model()

    def test_process_next_crops_empty_queue(self):
        self.input_queue.get.side_effect = mp.queues.Empty
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
        self.assertEqual(results, [])

    @patch("services.emotion_classification.cv2.cvtColor")
    @patch("services.emotion_classification.Image.fromarray")
    def test_classify_emotions_success(self, mock_fromarray, mock_cvtcolor):
        # Setup: Ensure model and transform are mocked or set appropriately
        self.emotion_classifier.model = MagicMock()
        self.emotion_classifier.transform = MagicMock(
            return_value=MagicMock(unsqueeze=MagicMock(return_value="input_tensor"))
        )
        self.emotion_classifier.device = "cpu"  # or mock torch.device

        mock_pil_image = MagicMock()
        mock_fromarray.return_value = mock_pil_image

        # Mock model output
        mock_output_tensor = torch.tensor([[0.1, 0.8, 0.1]])  # happy
        self.emotion_classifier.model.return_value = mock_output_tensor

        # Mock torch.nn.functional.softmax and torch.argmax
        with (
            patch(
                "services.emotion_classification.torch.nn.functional.softmax"
            ) as mock_softmax,
            patch("services.emotion_classification.torch.argmax") as mock_argmax,
        ):
            mock_softmax.return_value = torch.tensor([0.1, 0.8, 0.1])  # Probabilities
            mock_argmax.return_value = torch.tensor([1])  # Index of 'happy'

            crop_image_np = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            cropped_images_data = [
                {"image": crop_image_np, "bbox": (10, 10, 60, 60), "class": "dog"}
            ]

            # Call the method
            results = self.emotion_classifier._classify_emotions(cropped_images_data)

            # Assertions
            mock_cvtcolor.assert_called_once_with(crop_image_np, cv2.COLOR_BGR2RGB)
            mock_fromarray.assert_called_once()  # Argument is the result of cvtColor
            self.emotion_classifier.transform.assert_called_once_with(mock_pil_image)
            self.emotion_classifier.model.assert_called_once_with("input_tensor")
            mock_softmax.assert_called_once()
            mock_argmax.assert_called_once()

            self.assertEqual(len(results), 1)
            self.assertEqual(
                results[0]["emotion"], "happy"
            )  # class_names = ["angry", "happy", "sad"]
            self.assertAlmostEqual(results[0]["confidence"], 0.8, places=5)
            self.assertEqual(results[0]["bbox"], (10, 10, 60, 60))
            self.assertEqual(results[0]["class"], "dog")

    def test_classify_emotions_empty_or_invalid_crop_image(self):
        self.emotion_classifier.model = MagicMock()  # Needs a model to proceed
        self.emotion_classifier.transform = MagicMock()

        # Test with None image
        cropped_images_none = [{"image": None, "bbox": (0, 0, 0, 0), "class": 0}]
        results_none = self.emotion_classifier._classify_emotions(cropped_images_none)
        self.assertEqual(results_none, [])

        # Test with zero-size image
        crop_image_zerosize = np.array([])
        cropped_images_zerosize = [
            {"image": crop_image_zerosize, "bbox": (0, 0, 0, 0), "class": 0}
        ]
        results_zerosize = self.emotion_classifier._classify_emotions(
            cropped_images_zerosize
        )
        self.assertEqual(results_zerosize, [])

        # Test with invalid shape (e.g., 2D grayscale, or 4 channels)
        crop_image_2d = np.zeros((50, 50), dtype=np.uint8)
        cropped_images_2d = [
            {"image": crop_image_2d, "bbox": (0, 0, 10, 10), "class": 0}
        ]
        results_2d = self.emotion_classifier._classify_emotions(cropped_images_2d)
        self.assertEqual(results_2d, [])

        crop_image_4channel = np.zeros((50, 50, 4), dtype=np.uint8)
        cropped_images_4channel = [
            {"image": crop_image_4channel, "bbox": (0, 0, 10, 10), "class": 0}
        ]
        results_4channel = self.emotion_classifier._classify_emotions(
            cropped_images_4channel
        )
        self.assertEqual(results_4channel, [])

    def test_run_method_calls_load_model_and_process_loop(self):
        # Mock methods called by run
        self.emotion_classifier._load_model = MagicMock()
        self.emotion_classifier._process_next_crops = MagicMock()

        # Configure running.is_set to run the loop once then exit
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


if __name__ == "__main__":
    unittest.main()
