import multiprocessing as mp
import multiprocessing.queues as mpq
import time
from typing import Callable

import cv2
import numpy as np

from config import (
    PROCESS_QUEUE_TIMEOUT,
    QUEUE_COPIER_JOIN_TIMEOUT,
    WINDOW_WAIT_TIME,
)

BOUNDING_BOX_COLOR = (0, 255, 0)
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 2
TEXT_Y_OFFSET = -10
MAX_GRID_COLUMNS = 3
QUEUE_COPIER_TIMEOUT = 0.1

RAW_FEED_WINDOW_NAME = "Raw Camera Feed"
DETECTION_WINDOW_NAME = "Pet Detection"
CROPPED_PETS_WINDOW_NAME = "Cropped Pets"


def view_raw_frames(running: Callable[[], bool], raw_frame_queue: mpq.Queue) -> None:
    if not running():
        print("Pipeline is not running")
        return

    print(f"Displaying raw camera feed in window '{RAW_FEED_WINDOW_NAME}'")
    print("Press 'q' to exit")

    try:
        while True:
            try:
                frame, _ = raw_frame_queue.get(timeout=PROCESS_QUEUE_TIMEOUT)
                cv2.imshow(RAW_FEED_WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except mp.queues.Empty:
                time.sleep(WINDOW_WAIT_TIME)
    finally:
        cv2.destroyAllWindows()


def view_detection_results(
    running: Callable[[], bool],
    detected_objects_queue: mpq.Queue,
    pet_classes: dict[str, int],
) -> None:
    if not running():
        print("Pipeline is not running")
        return

    print(f"Displaying detection results in window '{DETECTION_WINDOW_NAME}'")
    print("Press 'q' to exit")

    try:
        while True:
            try:
                frame, _, detections = detected_objects_queue.get(
                    timeout=PROCESS_QUEUE_TIMEOUT
                )

                vis_frame = frame.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection["bbox"]
                    cv2.rectangle(
                        vis_frame,
                        (x1, y1),
                        (x2, y2),
                        BOUNDING_BOX_COLOR,
                        TEXT_THICKNESS,
                    )

                    class_id = detection["class"]
                    class_name = next(
                        (k for k, v in pet_classes.items() if v == class_id),
                        "animal",
                    )
                    label = f"{class_name}: {detection['confidence']:.2f}"
                    cv2.putText(
                        vis_frame,
                        label,
                        (x1, y1 + TEXT_Y_OFFSET),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        TEXT_FONT_SCALE,
                        BOUNDING_BOX_COLOR,
                        TEXT_THICKNESS,
                    )

                cv2.imshow(DETECTION_WINDOW_NAME, vis_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except mp.queues.Empty:
                time.sleep(WINDOW_WAIT_TIME)
    finally:
        cv2.destroyAllWindows()


def view_cropped_pets(
    running: Callable[[], bool],
    cropped_pets_queue: mpq.Queue,
) -> None:
    if not running():
        print("Pipeline is not running")
        return

    print(f"Displaying cropped pets in window '{CROPPED_PETS_WINDOW_NAME}'")
    print("Press 'q' to exit")

    try:
        while True:
            try:
                _, _, cropped_images = cropped_pets_queue.get(
                    timeout=PROCESS_QUEUE_TIMEOUT
                )

                if cropped_images:
                    display_grid = create_image_grid(cropped_images)
                    cv2.imshow(CROPPED_PETS_WINDOW_NAME, display_grid)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except mp.queues.Empty:
                time.sleep(WINDOW_WAIT_TIME)
    finally:
        cv2.destroyAllWindows()


def create_image_grid(cropped_images: list[dict[str, any]]) -> np.ndarray:
    max_width = max([img["image"].shape[1] for img in cropped_images])
    max_height = max([img["image"].shape[0] for img in cropped_images])

    cols = min(MAX_GRID_COLUMNS, len(cropped_images))
    rows = (len(cropped_images) + cols - 1) // cols

    grid = np.zeros((rows * max_height, cols * max_width, 3), dtype=np.uint8)

    for i, crop_data in enumerate(cropped_images):
        row, col = i // cols, i % cols
        crop = crop_data["image"]
        h, w = crop.shape[:2]

        y_start = row * max_height
        x_start = col * max_width
        grid[y_start : y_start + h, x_start : x_start + w] = crop

    return grid


def queue_copier(
    source_queue: mpq.Queue, dest_queues: list[mpq.Queue], running_event
) -> None:
    while running_event.is_set():
        try:
            item = source_queue.get(timeout=QUEUE_COPIER_TIMEOUT)
            for queue in dest_queues:
                queue.put(item)
        except mp.queues.Empty:
            continue


def start_queue_copier(
    source_queue: mpq.Queue, dest_queues: list[mpq.Queue], running_event
) -> mp.Process:
    copier_process = mp.Process(
        target=queue_copier,
        args=(source_queue, dest_queues, running_event),
    )
    copier_process.start()
    return copier_process


def process_raw_frames(raw_frames_copy: mpq.Queue) -> bool:
    try:
        raw_frame, _ = raw_frames_copy.get_nowait()
        with open(f"raw_frame_{time.time()}.jpg", "wb") as f:
            f.write(raw_frame)
        cv2.imshow(RAW_FEED_WINDOW_NAME, raw_frame)
        return True
    except mp.queues.Empty:
        return False


def process_detections(detections_copy: mpq.Queue, pet_classes: dict[str, int]) -> bool:
    try:
        frame, _, detections = detections_copy.get_nowait()
        vis_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            cv2.rectangle(
                vis_frame, (x1, y1), (x2, y2), BOUNDING_BOX_COLOR, TEXT_THICKNESS
            )

            class_id = detection["class"]
            class_name = next(
                (k for k, v in pet_classes.items() if v == class_id),
                "animal",
            )
            label = f"{class_name}: {detection['confidence']:.2f}"
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 + TEXT_Y_OFFSET),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_FONT_SCALE,
                BOUNDING_BOX_COLOR,
                TEXT_THICKNESS,
            )

        cv2.imshow(DETECTION_WINDOW_NAME, vis_frame)
        return True
    except mp.queues.Empty:
        return False


def process_crops(crops_copy: mpq.Queue) -> bool:
    try:
        _, _, cropped_images = crops_copy.get_nowait()

        if cropped_images:
            display_grid = create_image_grid(cropped_images)
            cv2.imshow(CROPPED_PETS_WINDOW_NAME, display_grid)
            return True
        return False
    except mp.queues.Empty:
        return False


def view_all_outputs(
    running: Callable[[], bool],
    queues: tuple[mpq.Queue, ...],
    pet_classes: dict[str, int],
) -> None:
    if not running():
        print("Pipeline is not running")
        return

    print("Displaying all pipeline outputs in separate windows")
    print("Press 'q' to exit")

    (
        raw_frame_queue,
        detected_objects_queue,
        cropped_pets_queue,
    ) = queues

    raw_frames_copy = mp.Queue()
    detections_copy = mp.Queue()
    crops_copy = mp.Queue()

    running_event = mp.Event()
    running_event.set()

    raw_copier = start_queue_copier(
        raw_frame_queue, [raw_frame_queue, raw_frames_copy], running_event
    )
    detection_copier = start_queue_copier(
        detected_objects_queue, [detected_objects_queue, detections_copy], running_event
    )
    crop_copier = start_queue_copier(cropped_pets_queue, [crops_copy], running_event)

    try:
        while True:
            # show_raw = process_raw_frames(raw_frames_copy)
            show_detection = process_detections(detections_copy, pet_classes)
            show_crops = process_crops(crops_copy)

            if not (show_detection or show_crops):
                time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        running_event.clear()
        raw_copier.join(timeout=QUEUE_COPIER_JOIN_TIMEOUT)
        detection_copier.join(timeout=QUEUE_COPIER_JOIN_TIMEOUT)
        crop_copier.join(timeout=QUEUE_COPIER_JOIN_TIMEOUT)

        cv2.destroyAllWindows()
