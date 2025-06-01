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
