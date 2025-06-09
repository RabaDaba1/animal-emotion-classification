from pathlib import Path

# Queue settings
RAW_FRAME_QUEUE_SIZE = 10
DETECTION_QUEUE_SIZE = 10
CROPPED_PETS_QUEUE_SIZE = 15
MAX_QUEUE_FILL_LEVEL = 5

# Process settings
PROCESS_JOIN_TIMEOUT = 3
QUEUE_COPIER_JOIN_TIMEOUT = 1

# Camera settings
DEFAULT_CAMERA_DEVICE_ID = 0
DEFAULT_FRAME_RATE = 1
CAMERA_SLEEP_TIME = 0.01

# Detection settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
YOLO_MODEL_NAME = "yolov10n.pt"
PROCESS_QUEUE_TIMEOUT = 1

# Image processing settings
BOUNDING_BOX_MARGIN_PERCENT = 0.05
WINDOW_WAIT_TIME = 0.01
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

# COCO class IDs for animal detection
PET_CLASSES = {
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
}

VISUALIZATION_QUEUE_SIZE = 5
SKIP_FRAMES_WHEN_BUSY = True

# Model settings
MODELS_DIR = Path("models")  # Relative to project root
EMOTION_MODEL_FILENAME = "mobilenetv3_pet_emotion_classifier.pth"
EMOTION_MODEL_CLASS_NAMES = ["angry", "happy", "sad"]
EMOTION_CLASSIFICATION_MODEL_PATH = MODELS_DIR / EMOTION_MODEL_FILENAME
YOLO_MODEL_PATH = MODELS_DIR / YOLO_MODEL_NAME

# Visualization settings
VIS_BOUNDING_BOX_COLOR = (0, 255, 0)  # Green
VIS_TEXT_FONT_SCALE = 0.5
VIS_TEXT_THICKNESS = 2
VIS_TEXT_Y_OFFSET = -10
VIS_MAX_GRID_COLUMNS = 3
VIS_QUEUE_COPIER_TIMEOUT = 0.1  # Timeout for queue copier get operation

VIS_RAW_FEED_WINDOW_NAME = "Raw Camera Feed"
VIS_DETECTION_WINDOW_NAME = "Pet Detection"
VIS_CROPPED_PETS_WINDOW_NAME = "Cropped Pets"
VIS_EMOTION_RESULTS_WINDOW_NAME = "Emotion Classification Results"
