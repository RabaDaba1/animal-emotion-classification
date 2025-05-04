import multiprocessing

from config import (
    CROPPED_PETS_QUEUE_SIZE,
    DEFAULT_CAMERA_DEVICE_ID,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAME_RATE,
    DETECTION_QUEUE_SIZE,
    RAW_FRAME_QUEUE_SIZE,
)
from services.image_acquisition import ImageAcquisition
from services.image_corpping import ImageCropping
from services.object_detection import ObjectDetection
from services.process_manager import ProcessManager

multiprocessing.freeze_support()

if __name__ == "__main__":
    raw_frame_queue = multiprocessing.Queue(maxsize=RAW_FRAME_QUEUE_SIZE)
    detected_objects_queue = multiprocessing.Queue(maxsize=DETECTION_QUEUE_SIZE)
    cropped_pets_queue = multiprocessing.Queue(maxsize=CROPPED_PETS_QUEUE_SIZE)
    emotion_results_queue = multiprocessing.Queue()

    image_acquisition = ImageAcquisition(
        output_queue=raw_frame_queue,
        frame_rate=DEFAULT_FRAME_RATE,
        device_id=DEFAULT_CAMERA_DEVICE_ID,
    )

    object_detection = ObjectDetection(
        input_queue=raw_frame_queue,
        output_queue=detected_objects_queue,
        confidence=DEFAULT_CONFIDENCE_THRESHOLD,
    )

    image_cropping = ImageCropping(
        input_queue=detected_objects_queue, output_queue=cropped_pets_queue
    )

    manager = ProcessManager(
        raw_frame_queue=raw_frame_queue,
        detected_objects_queue=detected_objects_queue,
        cropped_pets_queue=cropped_pets_queue,
        emotion_results_queue=emotion_results_queue,
        image_acquisition=image_acquisition,
        object_detection=object_detection,
        image_cropping=image_cropping,
    )

    manager.start()

    try:
        manager.view_all_outputs()

        # manager.view_raw_frames()
        # manager.view_detection_results()
        # manager.view_cropped_pets()
    finally:
        manager.stop()
