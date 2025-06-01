import multiprocessing as mp

import visualization as vis
from config import PROCESS_JOIN_TIMEOUT
from services.emotion_classification import EmotionClassification
from services.image_acquisition import ImageAcquisition
from services.image_corpping import ImageCropping
from services.object_detection import ObjectDetection


class ProcessManager:
    def __init__(
        self,
        raw_frame_queue: mp.Queue,
        detected_objects_queue: mp.Queue,
        cropped_pets_queue: mp.Queue,
        emotion_results_queue: mp.Queue,
        image_acquisition: ImageAcquisition,
        object_detection: ObjectDetection,
        image_cropping: ImageCropping,
        emotion_classification: EmotionClassification,
    ):
        self.raw_frame_queue = raw_frame_queue
        self.detected_objects_queue = detected_objects_queue
        self.cropped_pets_queue = cropped_pets_queue
        self.emotion_results_queue = emotion_results_queue

        self.image_acquisition = image_acquisition
        self.object_detection = object_detection
        self.image_cropping = image_cropping
        self.emotion_classification = emotion_classification

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

        print("Starting emotion classification process...")
        self.emotion_classification.start()

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
        self.emotion_classification.stop()

        self.image_acquisition.join(timeout=PROCESS_JOIN_TIMEOUT)
        self.object_detection.join(timeout=PROCESS_JOIN_TIMEOUT)
        self.image_cropping.join(timeout=PROCESS_JOIN_TIMEOUT)
        self.emotion_classification.join(timeout=PROCESS_JOIN_TIMEOUT)

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

    def view_emotion_results(self):
        vis.view_emotion_results(self.is_running, self.emotion_results_queue)

    def view_all_outputs(self):
        queues = (
            self.raw_frame_queue,
            self.detected_objects_queue,
            self.cropped_pets_queue,
            self.emotion_results_queue,
        )
        vis.view_all_outputs(self.is_running, queues, self.object_detection.pet_classes)
