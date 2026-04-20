import numpy as np
import cv2
import mss
import time
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from evdev import ecodes as e
import config
from utils import ui, all_keys_up

class LinuxTMNFEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.sct = mss.mss()
        
        # 3 Actions: 0: Straight, 1: Left, 2: Right
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(72, 96, 1), dtype=np.uint8)

        self.history = deque(maxlen=config.STUCK_WINDOW)
        self.frames_since_reset = 0
        self.low_speed_timer = 0
        self.is_menu_locked = False
        self.total_finishes = 0
        self.just_updated = False

        self.prev_speed = 0.0
        self.speed_ema = 0.0
        self.ema_alpha = 0.30
        self.last_steer_action = 0

    def _get_obs_metrics(self):
        sct_img = self.sct.grab(config.MONITOR)
        raw = np.array(sct_img)[:, :, :3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        # Detection logic
        finish_roi = raw[320:480, 170:470]
        hsv_finish = cv2.cvtColor(finish_roi, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_finish, np.array([0, 150, 100]), np.array([10, 255, 255]))
        red_count = np.count_nonzero(red_mask)

        roi_grass = raw[240:, :, :]
        hsv_grass = cv2.cvtColor(roi_grass, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_grass, np.array([35, 50, 50]), np.array([90, 255, 255]))
        green_count = np.count_nonzero(green_mask)

        speed_roi = gray[448:473, 540:620]
        _, thresh = cv2.threshold(speed_roi, 220, 255, cv2.THRESH_BINARY)
        speed = float(np.sum(thresh) // 1020)

        obs_resized = cv2.resize(gray, (96, 72), interpolation=cv2.INTER_NEAREST)
        wall_check_area = obs_resized[25:50, 30:65]

        return np.expand_dims(obs_resized, axis=-1), speed, red_count, green_count, wall_check_area

    def reset(self, seed=None, options=None):
        all_keys_up()
        if not self.is_menu_locked:
            ui.write(e.EV_KEY, config.KEYS["BACK"], 1); ui.syn(); time.sleep(0.1)
            ui.write(e.EV_KEY, config.KEYS["BACK"], 0); ui.syn(); time.sleep(1.2)

        self.history.clear()
        self.frames_since_reset = 0
        self.low_speed_timer = 0
        obs, speed, _, _, _ = self._get_obs_metrics()
        self.speed_ema = speed
        return obs, {}

    def step(self, action):
        if self.is_menu_locked:
            return np.zeros((72, 96, 1), dtype=np.uint8), 0.0, False, False, {}

        obs, speed, red_count, green_count, wall_area = self._get_obs_metrics()

        # --- THE FIX: FORCED GAS ---
        ui.write(e.EV_KEY, config.KEYS["LEFT"], 0)
        ui.write(e.EV_KEY, config.KEYS["RIGHT"], 0)
        
        # Explicitly hold Gas every frame
        ui.write(e.EV_KEY, config.KEYS["UP"], 1)

        steer_dir = 0
        if action == 1: 
            ui.write(e.EV_KEY, config.KEYS["LEFT"], 1)
            steer_dir = -1
        elif action == 2: 
            ui.write(e.EV_KEY, config.KEYS["RIGHT"], 1)
            steer_dir = 1
        
        ui.syn()

        # Debug print
        print(f"GAS: ON | ACT: {action} | SPD: {speed:.1f}", end='\r')

        # --- Rewards ---
        self.frames_since_reset += 1
        self.speed_ema = self.ema_alpha * speed + (1.0 - self.ema_alpha) * self.speed_ema
        reward = np.clip(self.speed_ema / 150.0, 0.0, 1.2)

        # Fail-safes
        done = False
        if green_count > config.GREEN_PIXEL_THRESHOLD: done = True
        if speed < 5 and self.frames_since_reset > 100: done = True

        return obs, float(reward), done, False, {}
