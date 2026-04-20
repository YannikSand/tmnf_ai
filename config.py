import evdev.ecodes as e

NUM_THREADS = 2
SAVE_PATH = "./models/tmnf_fresh_start_new"

STUCK_SIMILARITY_LIMIT = 0.9700
STUCK_WINDOW = 35
MIN_SPEED_THRESHOLD = 5.0
RED_PIXEL_THRESHOLD = 8000
GREEN_PIXEL_THRESHOLD = 1200

START_GRACE_PERIOD = 150
OFFTRACK_GRACE_PERIOD = 300
MAX_RACE_FRAMES = 3600

# CHANGED: Using W, A, D (standard codes 17, 30, 32)
KEYS = {
    "UP": 17,    # KEY_W
    "LEFT": 30,  # KEY_A
    "RIGHT": 32, # KEY_D
    "BACK": 14,  # KEY_BACKSPACE
    "ENTER": 28  # KEY_ENTER
}

MONITOR = {"top": 40, "left": 25, "width": 640, "height": 480}