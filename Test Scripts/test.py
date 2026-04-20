import cv2
import mss
import numpy as np
import time
from evdev import UInput, ecodes as e

# --- Settings ---
FINISH_BRIGHTNESS_THRESHOLD = 12.0
GREEN_PIXEL_THRESHOLD = 120000 
STUCK_DIFF_THRESHOLD = 0.5   # Sensitivity for wall freeze
STUCK_FRAME_LIMIT = 30       # 3 seconds at 0.1s sleep

# --- Hardware Input Setup ---
ui = UInput()
K_ENTER = e.KEY_ENTER
K_BACK = e.KEY_BACKSPACE

# Speed ROI (Coordinates from your main script)
SX, SY, SW, SH = 540, 448, 80, 25 

def press_enter_sequence():
    """Restarts after a finish screen."""
    print("[ACTION] FINISH DETECTED -> Menu Navigation...")
    ui.write(e.EV_KEY, K_ENTER, 1); ui.syn()
    time.sleep(0.1)
    ui.write(e.EV_KEY, K_ENTER, 0); ui.syn()
    time.sleep(1.2) 
    ui.write(e.EV_KEY, K_ENTER, 1); ui.syn()
    time.sleep(0.1)
    ui.write(e.EV_KEY, K_ENTER, 0); ui.syn()

def quick_reset(reason):
    """Restarts after a crash or grass."""
    print(f"[ACTION] RESETTING: {reason}")
    ui.write(e.EV_KEY, K_BACK, 1); ui.syn()
    time.sleep(0.1)
    ui.write(e.EV_KEY, K_BACK, 0); ui.syn()

# --- Setup Screen Capture ---
sct = mss.mss()
monitor = {"top": 40, "left": 25, "width": 640, "height": 480}

print("--- HYBRID AUTO-RESETTER STARTING ---")
print("I will handle Finish, Walls, and Grass while you drive.")

prev_speed_roi = None
stuck_counter = 0

try:
    while True:
        sct_img = sct.grab(monitor)
        raw = np.array(sct_img)[:, :, :3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        # 1. FINISH DETECTION (Highest Priority)
        timer_area = raw[440:470, 300:400]
        avg_brightness = np.mean(timer_area)
        
        if avg_brightness < FINISH_BRIGHTNESS_THRESHOLD:
            press_enter_sequence()
            stuck_counter = 0
            time.sleep(4.0) # Reload cooldown
            continue

        # 2. GRASS DETECTION
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50]); upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_px = np.sum(green_mask > 0)

        if green_px > GREEN_PIXEL_THRESHOLD:
            quick_reset("Grass")
            stuck_counter = 0
            time.sleep(1.5)
            continue

        # 3. FUZZY WALL DETECTION
        speed_roi = gray[SY:SY+SH, SX:SX+SW]
        if prev_speed_roi is not None:
            # Calculate difference (MSE)
            err = np.sum((speed_roi.astype("float") - prev_speed_roi.astype("float")) ** 2)
            err /= float(speed_roi.shape[0] * speed_roi.shape[1])
            
            if err < STUCK_DIFF_THRESHOLD:
                stuck_counter += 1
            else:
                stuck_counter = 0

            if stuck_counter > STUCK_FRAME_LIMIT:
                quick_reset("Wall Stuck")
                stuck_counter = 0
                time.sleep(1.5)
                continue

        prev_speed_roi = speed_roi
        time.sleep(0.1) # 10 Checks per second

except KeyboardInterrupt:
    print("\nStopped.")