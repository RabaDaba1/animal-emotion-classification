import multiprocessing as mp
import multiprocessing.queues as mpq

import cv2
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image

from config import PROCESS_QUEUE_TIMEOUT


class EmotionClassification(mp.Process):
    def __init__(self, input_queue: mpq.Queue, output_queue: mpq.Queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = mp.Event()

        # Model will be loaded in the process
        self.model = None
        self.transform = None
        self.class_names = ["angry", "happy", "sad"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        self.running.set()
        self._load_model()

        try:
            while self.running.is_set():
                self._process_next_crops()
        except Exception as e:
            print(f"Error in EmotionClassification: {e}")

    def _load_model(self):
        """Load the trained emotion classification model"""
        try:
            # For now, create a dummy model - replace with actual model loading
            self.model = timm.create_model(
                "mobilenetv3_large_100", pretrained=True, num_classes=3
            )
            self.model.to(self.device)
            self.model.eval()

            # Define transform
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            print("Emotion classification model loaded successfully")
        except Exception as e:
            print(f"Failed to load emotion classification model: {e}")
            self.model = None

    def _process_next_crops(self):
        try:
            frame, timestamp, cropped_images = self.input_queue.get(
                timeout=PROCESS_QUEUE_TIMEOUT
            )
            if cropped_images:
                emotion_results = self._classify_emotions(cropped_images)
                if emotion_results:
                    try:
                        self.output_queue.put_nowait(
                            (frame, timestamp, emotion_results)
                        )
                    except Exception:
                        pass
        except mp.queues.Empty:
            pass
        except Exception as e:
            print(f"Emotion classification error: {e}")

    def _classify_emotions(self, cropped_images: list[dict]) -> list[dict]:
        """Classify emotions for all cropped images"""
        if self.model is None:
            return []

        emotion_results = []

        for crop_data in cropped_images:
            crop_image = crop_data["image"]

            if crop_image is None or crop_image.size == 0:
                continue

            try:
                if len(crop_image.shape) != 3 or crop_image.shape[2] != 3:
                    continue

                # Convert to PIL Image
                if crop_image.shape[2] == 3:  # BGR to RGB
                    crop_image_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                else:
                    crop_image_rgb = crop_image

                pil_image = Image.fromarray(crop_image_rgb)

                # Apply transform and predict
                input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[predicted_class].item()

                emotion = self.class_names[predicted_class]

                emotion_results.append(
                    {
                        "emotion": emotion,
                        "confidence": confidence,
                        "bbox": crop_data["bbox"],
                        "class": crop_data["class"],
                    }
                )

            except Exception as e:
                print(f"Error classifying emotion for single crop: {e}")
                continue

        return emotion_results

    def stop(self):
        self.running.clear()
