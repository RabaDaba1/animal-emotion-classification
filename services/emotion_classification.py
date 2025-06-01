import multiprocessing as mp
import multiprocessing.queues as mpq
import os
from typing import Optional

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from config import PROCESS_QUEUE_TIMEOUT


class EmotionClassification(mp.Process):
    def __init__(
        self,
        input_queue: mpq.Queue,
        output_queue: mpq.Queue,
        model_path: Optional[str] = None,
    ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_path = model_path or self._find_model_path()
        self.running = mp.Event()

        # Model will be loaded in the child process
        self.model = None
        self.device = None
        self.transform = None
        self.class_names = ["angry", "happy", "sad"]

    def _find_model_path(self) -> str:
        """Find the model file in the project directory"""
        project_root = os.path.dirname(os.path.dirname(__file__))
        model_file = "mobilenetv3_pet_emotion_classifier.pth"

        # Check common locations
        possible_paths = [
            os.path.join(project_root, model_file),
            os.path.join(project_root, "models", model_file),
            os.path.join(project_root, "pet_emotion_model", model_file),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Model file {model_file} not found in expected locations"
        )

    def _load_model(self):
        """Load the trained emotion classification model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Create model architecture
            self.model = timm.create_model(
                "mobilenetv3_large_100", pretrained=False, num_classes=3
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            # Get class names from checkpoint if available
            if "class_names" in checkpoint:
                self.class_names = checkpoint["class_names"]

            # Define transform for inference
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            print(f"Emotion classification model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading emotion classification model: {e}")
            self.running.clear()

    def run(self):
        self.running.set()
        self._load_model()

        if self.model is None:
            return

        try:
            while self.running.is_set():
                self._process_next_batch()
        except Exception as e:
            print(f"Error in EmotionClassification: {e}")

    def _process_next_batch(self):
        try:
            frame, timestamp, cropped_images = self.input_queue.get(
                timeout=PROCESS_QUEUE_TIMEOUT
            )
            emotion_results = self._classify_emotions(cropped_images)
            self.output_queue.put((frame, timestamp, emotion_results))
        except mp.queues.Empty:
            pass

    def _classify_emotions(self, cropped_images: list[dict]) -> list[dict]:
        """Classify emotions for all cropped images"""
        results = []

        for crop_data in cropped_images:
            try:
                emotion_pred = self._classify_single_image(crop_data["image"])

                result = {
                    "image": crop_data["image"],
                    "bbox": crop_data["bbox"],
                    "animal_class": crop_data["class"],
                    "animal_confidence": crop_data["confidence"],
                    "emotion": emotion_pred["emotion"],
                    "emotion_confidence": emotion_pred["confidence"],
                    "emotion_probabilities": emotion_pred["probabilities"],
                }
                results.append(result)

            except Exception as e:
                print(f"Error classifying emotion for image: {e}")
                # Add result with unknown emotion
                result = {
                    "image": crop_data["image"],
                    "bbox": crop_data["bbox"],
                    "animal_class": crop_data["class"],
                    "animal_confidence": crop_data["confidence"],
                    "emotion": "unknown",
                    "emotion_confidence": 0.0,
                    "emotion_probabilities": [0.0, 0.0, 0.0],
                }
                results.append(result)

        return results

    def _classify_single_image(self, image: np.ndarray) -> dict:
        """Classify emotion for a single image"""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Apply transforms
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[predicted_class].item()

        return {
            "emotion": self.class_names[predicted_class],
            "confidence": confidence,
            "probabilities": probabilities.cpu().numpy().tolist(),
        }

    def stop(self):
        self.running.clear()
