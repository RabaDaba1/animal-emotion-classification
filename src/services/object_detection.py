import multiprocessing as mp
import multiprocessing.queues as mpq

from ultralytics import YOLO

from src.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    MODELS_DIR,
    PET_CLASSES,
    PROCESS_QUEUE_TIMEOUT,
    YOLO_MODEL_NAME,
)


class ObjectDetection(mp.Process):
    def __init__(
        self,
        input_queue: mpq.Queue,
        output_queue: mpq.Queue,
        confidence=DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.confidence = confidence
        self.running = mp.Event()

        self.pet_classes = PET_CLASSES
        self.yolo_model = YOLO(MODELS_DIR / YOLO_MODEL_NAME)

    def run(self):
        self.running.set()
        try:
            while self.running.is_set():
                self._process_next_frame()
        except Exception as e:
            print(f"Error in ObjectDetection: {e}")

    def _process_next_frame(self):
        try:
            frame, timestamp = self.input_queue.get(timeout=PROCESS_QUEUE_TIMEOUT)
            detections = self._detect_pets(frame)
            try:
                self.output_queue.put_nowait((frame, timestamp, detections))
            except:
                pass
        except mp.queues.Empty:
            pass
        except Exception as e:
            print(f"Frame processing error: {e}")

    def _detect_pets(self, frame):
        try:
            results = self.yolo_model(frame, conf=self.confidence)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls.item())
                        if cls in self.pet_classes.values():
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = box.conf.item()
                            detections.append(
                                {
                                    "bbox": (x1, y1, x2, y2),
                                    "confidence": conf,
                                    "class": cls,
                                }
                            )

            return detections
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def stop(self):
        self.running.clear()
