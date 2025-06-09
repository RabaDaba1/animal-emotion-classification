import multiprocessing as mp
from unittest.mock import MagicMock, patch

import numpy as np

from src.config import BOUNDING_BOX_MARGIN_PERCENT, PROCESS_QUEUE_TIMEOUT
from src.services.image_cropping import ImageCropping


class TestImageCropping:
    def setup_method(self, method):
        self.input_queue = MagicMock(spec=type(mp.Queue()))  # Use type(mp.Queue())
        self.output_queue = MagicMock(spec=type(mp.Queue()))  # Use type(mp.Queue())
        self.image_cropper = ImageCropping(self.input_queue, self.output_queue)
        self.image_cropper.running = MagicMock(spec=type(mp.Event()))

    def teardown_method(self, method):
        if hasattr(self.image_cropper, "running") and isinstance(
            self.image_cropper.running, MagicMock
        ):
            self.image_cropper.running.is_set.reset_mock()
            self.image_cropper.running.clear.reset_mock()
            self.image_cropper.running.set.reset_mock()

    def test_process_next_detections_empty_queue(self):
        self.input_queue.get.side_effect = mp.queues.Empty  # This should be fine
        self.image_cropper._process_next_detections()
        self.input_queue.get.assert_called_once_with(timeout=PROCESS_QUEUE_TIMEOUT)
        self.output_queue.put_nowait.assert_not_called()

    @patch.object(ImageCropping, "_crop_detections")
    def test_process_next_detections_success(self, mock_crop_detections):
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_timestamp = "2023-01-01 12:00:00"
        mock_detections = [{"bbox": (10, 10, 30, 30), "class": 0, "confidence": 0.9}]
        mock_cropped_results = [
            {
                "image": np.zeros((20, 20, 3)),
                "class": 0,
                "confidence": 0.9,
                "bbox": (10, 10, 30, 30),
            }
        ]

        self.input_queue.get.return_value = (
            mock_frame,
            mock_timestamp,
            mock_detections,
        )
        mock_crop_detections.return_value = mock_cropped_results

        self.image_cropper._process_next_detections()

        self.input_queue.get.assert_called_once_with(timeout=PROCESS_QUEUE_TIMEOUT)
        mock_crop_detections.assert_called_once_with(mock_frame, mock_detections)
        self.output_queue.put_nowait.assert_called_once_with(
            (mock_frame, mock_timestamp, mock_cropped_results)
        )

    @patch.object(ImageCropping, "_crop_detections")
    def test_process_next_detections_no_cropped_images(self, mock_crop_detections):
        mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_timestamp = "2023-01-01 12:00:00"
        mock_detections = [{"bbox": (10, 10, 30, 30), "class": 0, "confidence": 0.9}]

        self.input_queue.get.return_value = (
            mock_frame,
            mock_timestamp,
            mock_detections,
        )
        mock_crop_detections.return_value = []  # No cropped images

        self.image_cropper._process_next_detections()
        self.output_queue.put_nowait.assert_not_called()

    def test_crop_single_detection_success(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = {"bbox": (20, 20, 80, 80)}  # 60x60 box

        # Expected crop size with margin (e.g., 10% margin on 60 is 6)
        # x1_margin = 20 - 6 = 14
        # y1_margin = 20 - 6 = 14
        # x2_margin = 80 + 6 = 86
        # y2_margin = 80 + 6 = 86
        # Expected height/width: 86-14 = 72

        expected_h = int(60 + 2 * (60 * BOUNDING_BOX_MARGIN_PERCENT))
        expected_w = int(60 + 2 * (60 * BOUNDING_BOX_MARGIN_PERCENT))

        cropped_image = self.image_cropper._crop_single_detection(frame, detection)
        assert cropped_image is not None
        assert cropped_image.shape[0] == expected_h  # Height
        assert cropped_image.shape[1] == expected_w  # Width
        assert cropped_image.shape[2] == 3  # Channels

    def test_crop_single_detection_invalid_bbox_zero_hw(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection_zero_h = {"bbox": (20, 20, 80, 20)}  # height = 0
        detection_zero_w = {"bbox": (20, 20, 20, 80)}  # width = 0

        assert (
            self.image_cropper._crop_single_detection(frame, detection_zero_h) is None
        )
        assert (
            self.image_cropper._crop_single_detection(frame, detection_zero_w) is None
        )

    def test_crop_single_detection_margins_out_of_bounds(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)  # Small frame
        # Bbox near edge, margin will go out of bounds
        detection_near_edge = {"bbox": (0, 0, 10, 10)}
        # x1=0, y1=0, x2=10, y2=10. h=10, w=10. margin = 10 * BBOX_MARGIN (e.g. 0.1 -> 1)
        # x1_margin = max(0, 0 - 1) = 0
        # y1_margin = max(0, 0 - 1) = 0
        # x2_margin = min(50, 10 + 1) = 11
        # y2_margin = min(50, 10 + 1) = 11
        # Expected shape: (11, 11, 3)

        cropped_image = self.image_cropper._crop_single_detection(
            frame, detection_near_edge
        )
        assert cropped_image is not None

        h, w = 10, 10
        margin_x = int(w * BOUNDING_BOX_MARGIN_PERCENT)
        margin_y = int(h * BOUNDING_BOX_MARGIN_PERCENT)

        expected_x1 = max(0, 0 - margin_x)
        expected_y1 = max(0, 0 - margin_y)
        expected_x2 = min(frame.shape[1], 10 + margin_x)
        expected_y2 = min(frame.shape[0], 10 + margin_y)

        assert cropped_image.shape[0] == expected_y2 - expected_y1
        assert cropped_image.shape[1] == expected_x2 - expected_x1

    def test_crop_single_detection_inverted_margins(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # A case where x2_margin could be <= x1_margin if logic was flawed (not expected with current code)
        # This test is more for robustness if margin calculation or bbox were unusual
        # e.g. if bbox was tiny and margin was negative or very large.
        # With current positive margin, this is hard to trigger unless bbox is outside frame initially.
        # Let's test a bbox that is already outside, which should be handled by max/min.
        detection_outside = {
            "bbox": (120, 120, 130, 130)
        }  # Completely outside 100x100 frame

        # x1_margin = max(0, 120 - margin)
        # y1_margin = max(0, 120 - margin)
        # x2_margin = min(100, 130 + margin)
        # y2_margin = min(100, 130 + margin)
        # If x1_margin ends up >= x2_margin, should return None.
        # Example: x1_margin = max(0, 120 - 1) = 119. x2_margin = min(100, 130 + 1) = 100.
        # Here 119 > 100, so x2_margin <= x1_margin is true.
        cropped_image = self.image_cropper._crop_single_detection(
            frame, detection_outside
        )
        assert cropped_image is None

    @patch.object(ImageCropping, "_crop_single_detection")
    def test_crop_detections(self, mock_crop_single):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections_data = [
            {"bbox": (10, 10, 20, 20), "class": 0, "confidence": 0.9},
            {"bbox": (30, 30, 40, 40), "class": 1, "confidence": 0.8},
        ]
        mock_cropped_img1 = np.zeros((10, 10, 3))
        mock_cropped_img2 = np.zeros((10, 10, 3))
        mock_crop_single.side_effect = [mock_cropped_img1, mock_cropped_img2]

        results = self.image_cropper._crop_detections(frame, detections_data)

        assert mock_crop_single.call_count == 2
        mock_crop_single.assert_any_call(frame, detections_data[0])
        mock_crop_single.assert_any_call(frame, detections_data[1])

        assert len(results) == 2
        assert results[0]["image"] is mock_cropped_img1
        assert results[0]["class"] == detections_data[0]["class"]
        assert results[0]["confidence"] == detections_data[0]["confidence"]
        assert results[0]["bbox"] == detections_data[0]["bbox"]

        assert results[1]["image"] is mock_cropped_img2
        assert results[1]["class"] == detections_data[1]["class"]

    def test_run_method_calls_process_loop(self):
        self.image_cropper._process_next_detections = MagicMock()
        self.image_cropper.running.is_set.side_effect = [True, False]  # Loop once

        self.image_cropper.run()

        self.image_cropper.running.set.assert_called_once()
        self.image_cropper._process_next_detections.assert_called_once()

    def test_run_method_handles_exception_in_loop(self):
        self.image_cropper._process_next_detections = MagicMock(
            side_effect=Exception("Test loop error")
        )
        self.image_cropper.running.is_set.side_effect = [True, False]  # Loop once

        with patch("builtins.print") as mock_print:
            self.image_cropper.run()

        self.image_cropper.running.set.assert_called_once()
        self.image_cropper._process_next_detections.assert_called_once()
        mock_print.assert_any_call("Error in ImageCropping: Test loop error")

    def test_stop(self):
        self.image_cropper.stop()
        self.image_cropper.running.clear.assert_called_once()
