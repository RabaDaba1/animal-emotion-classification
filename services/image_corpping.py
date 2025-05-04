import multiprocessing as mp
import multiprocessing.queues as mpq

import numpy as np

from config import BOUNDING_BOX_MARGIN_PERCENT, PROCESS_QUEUE_TIMEOUT


class ImageCropping(mp.Process):
    def __init__(self, input_queue: mpq.Queue, output_queue: mpq.Queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = mp.Event()

    def run(self):
        self.running.set()

        try:
            while self.running.is_set():
                self._process_next_detections()
        except Exception as e:
            print(f"Error in ImageCropping: {e}")

    def _process_next_detections(self):
        try:
            frame, timestamp, detections = self.input_queue.get(
                timeout=PROCESS_QUEUE_TIMEOUT
            )
            cropped_images = self._crop_detections(frame, detections)

            if cropped_images:
                self.output_queue.put((frame, timestamp, cropped_images))
        except mp.queues.Empty:
            pass

    def _crop_detections(
        self, frame: np.ndarray, detections: dict[str, any]
    ) -> list[dict[str, any]]:
        cropped_images = []
        for detection in detections:
            cropped = self._crop_single_detection(frame, detection)
            cropped_images.append(
                {
                    "image": cropped,
                    "class": detection["class"],
                    "confidence": detection["confidence"],
                    "bbox": detection["bbox"],
                }
            )
        return cropped_images

    def _crop_single_detection(
        self, frame: np.ndarray, detection: dict[str, any]
    ) -> np.ndarray:
        x1, y1, x2, y2 = detection["bbox"]
        h, w = y2 - y1, x2 - x1
        margin_x, margin_y = (
            int(w * BOUNDING_BOX_MARGIN_PERCENT),
            int(h * BOUNDING_BOX_MARGIN_PERCENT),
        )

        x1_margin = max(0, x1 - margin_x)
        y1_margin = max(0, y1 - margin_y)
        x2_margin = min(frame.shape[1], x2 + margin_x)
        y2_margin = min(frame.shape[0], y2 + margin_y)

        return frame[y1_margin:y2_margin, x1_margin:x2_margin]

    def stop(self):
        self.running.clear()
