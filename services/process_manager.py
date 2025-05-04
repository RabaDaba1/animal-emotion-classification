import multiprocessing as mp

import visualization as vis
from config import (
    CROPPED_PETS_QUEUE_SIZE,
    DETECTION_QUEUE_SIZE,
    PROCESS_JOIN_TIMEOUT,
    RAW_FRAME_QUEUE_SIZE,
)
from services.image_acquisition import ImageAcquisition
from services.image_corpping import ImageCropping
from services.object_detection import ObjectDetection


class ProcessManager:
    def __init__(self):
        self.raw_frame_queue = mp.Queue(maxsize=RAW_FRAME_QUEUE_SIZE)
        self.detected_objects_queue = mp.Queue(maxsize=DETECTION_QUEUE_SIZE)
        self.cropped_pets_queue = mp.Queue(maxsize=CROPPED_PETS_QUEUE_SIZE)
        self.emotion_results_queue = mp.Queue()

        self.image_acquisition = ImageAcquisition(output_queue=self.raw_frame_queue)
        self.object_detection = ObjectDetection(
            input_queue=self.raw_frame_queue, output_queue=self.detected_objects_queue
        )
        self.image_cropping = ImageCropping(
            input_queue=self.detected_objects_queue,
            output_queue=self.cropped_pets_queue,
        )

        # TODO: self.emotion_classification

        self.running = False

    def start(self):
        if self.running:
            print("Pipeline is already running")
            return

        print("Starting image acquisition process...")
        self.image_acquisition.start()

        print("Starting object detection process...")
        self.object_detection.start()

        print("Starting image cropping process...")
        self.image_cropping.start()

        # TODO: Emotion classification

        self.running = True
        print("Pipeline started successfully")

    def stop(self):
        if not self.running:
            print("Pipeline is not running")
            return

        print("Stopping processes...")
        self._stop_processes()
        self._clear_queues()

        self.running = False
        print("Pipeline stopped")

    def _stop_processes(self):
        self.image_acquisition.stop()
        self.object_detection.stop()
        self.image_cropping.stop()

        self.image_acquisition.join(timeout=PROCESS_JOIN_TIMEOUT)
        self.object_detection.join(timeout=PROCESS_JOIN_TIMEOUT)
        self.image_cropping.join(timeout=PROCESS_JOIN_TIMEOUT)

    def _clear_queues(self):
        self._clear_queue(self.raw_frame_queue)
        self._clear_queue(self.detected_objects_queue)
        self._clear_queue(self.cropped_pets_queue)
        self._clear_queue(self.emotion_results_queue)

    def _clear_queue(self, queue: mp.Queue):
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                pass

    def is_running(self):
        return self.running

    def view_raw_frames(self):
        vis.view_raw_frames(self.is_running, self.raw_frame_queue)

    def view_detection_results(self):
        vis.view_detection_results(
            self.is_running,
            self.detected_objects_queue,
            self.object_detection.pet_classes,
        )

    def view_cropped_pets(self):
        vis.view_cropped_pets(self.is_running, self.cropped_pets_queue)

    def view_all_outputs(self):
        queues = (
            self.raw_frame_queue,
            self.detected_objects_queue,
            self.cropped_pets_queue,
        )
        vis.view_all_outputs(self.is_running, queues, self.object_detection.pet_classes)
