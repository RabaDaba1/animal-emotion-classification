import multiprocessing as mp
import multiprocessing.queues as mpq
import time
from datetime import datetime

import cv2

from src.config import (
    CAMERA_SLEEP_TIME,
    DEFAULT_CAMERA_DEVICE_ID,
    DEFAULT_FRAME_RATE,
    MAX_QUEUE_FILL_LEVEL,
)


class ImageAcquisition(mp.Process):
    def __init__(
        self,
        output_queue: mpq.Queue,
        frame_rate: int = DEFAULT_FRAME_RATE,
        device_id: int = DEFAULT_CAMERA_DEVICE_ID,
    ):
        super().__init__()
        self.output_queue = output_queue
        self.frame_rate = frame_rate
        self.device_id = device_id
        self.running = mp.Event()

    def run(self):
        self.running.set()
        camera = self._setup_camera()
        if camera is None:
            return

        delay = 1.0 / self.frame_rate
        last_time = time.time()

        try:
            while self.running.is_set():
                current_time = time.time()
                if current_time - last_time >= delay:
                    self._capture_frame(camera)
                    last_time = current_time
                else:
                    time.sleep(CAMERA_SLEEP_TIME)
        except Exception as e:
            print(f"Error in ImageAcquisition: {e}")
        finally:
            camera.release()

    def _setup_camera(self) -> cv2.VideoCapture:
        camera = cv2.VideoCapture(self.device_id)
        if not camera.isOpened():
            print("Error: Could not open camera.")
            self.running.clear()
            return None
        return camera

    def _capture_frame(self, camera: cv2.VideoCapture) -> None:
        try:
            ret, frame = camera.read()
            if not ret:
                print("Error: Failed to capture image")
                return

            timestamp = datetime.now()
            if self.output_queue.qsize() < MAX_QUEUE_FILL_LEVEL:
                try:
                    self.output_queue.put_nowait((frame, timestamp))
                except Exception:
                    pass
        except Exception as e:
            print(f"Frame capture error: {e}")

    def stop(self):
        self.running.clear()
