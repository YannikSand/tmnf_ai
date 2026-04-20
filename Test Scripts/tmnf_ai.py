import os
import sys
import numpy as np
import cv2
import mss
import time
import torch
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from evdev import UInput, ecodes as e
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from pynput import keyboard

# --- LAPTOP OPTIMIZATIONS ---
torch.set_num_threads(2) 

# --- Pathing ---
save_path = "./models/tmnf_fresh_start"

# --- Thresholds & Rewards ---
STUCK_SIMILARITY_LIMIT = 0.9700
STUCK_WINDOW = 35              
MIN_SPEED_THRESHOLD = 5.0      
START_GRACE_PERIOD = 150       
OFFTRACK_GRACE_PERIOD = 300   
RED_PIXEL_THRESHOLD = 8000     
GREEN_PIXEL_THRESHOLD = 120000 
MAX_RACE_FRAMES = 3600         

ui = UInput()
K_UP, K_LEFT, K_RIGHT, K_BACK, K_ENTER = e.KEY_UP, e.KEY_LEFT, e.KEY_RIGHT, e.KEY_BACKSPACE, e.KEY_ENTER

# Global state trackers
JUST_UPDATED = False
TOTAL_FINISHES = 0
TOTAL_RESETS = 0

def all_keys_up():
    for k in [K_UP, K_LEFT, K_RIGHT, K_BACK, K_ENTER]:
        ui.write(e.EV_KEY, k, 0)
    ui.syn()

class TMNFCallback(BaseCallback):
    def __init__(self, check_freq=2048):
        super(TMNFCallback, self).__init__()
        self.stop_training = False
        self.check_freq = check_freq
        self.start_time = time.time()
        self.step_counter = 0
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == 'q':
                print("\n[STOP] Saving model and exiting...")
                self.stop_training = True
        except AttributeError: pass

    def _on_step(self) -> bool:
        # If the environment is busy clearing menus, we EXIT this function immediately.
        # This prevents 'n_calls' from incrementing, effectively pausing the Brain.
        if self.training_env.get_attr("is_menu_locked")[0]:
            return True 

        self.step_counter += 1
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            fps = self.step_counter / max(elapsed, 1e-3)
            print(f">>> [STATS] Step: {self.num_timesteps} | Drive FPS: {fps:.2f} | Finishes: {TOTAL_FINISHES}")
            self.start_time = time.time()
            self.step_counter = 0
        return not self.stop_training

    def _on_rollout_end(self):
        global JUST_UPDATED
        all_keys_up()
        JUST_UPDATED = True 

class LinuxTMNFEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.sct = mss.mss()
        self.monitor = {"top": 40, "left": 25, "width": 640, "height": 480}
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, shape=(72, 96, 1), dtype=np.uint8)
        self.history = deque(maxlen=STUCK_WINDOW)
        self.frames_since_reset = 0
        self.low_speed_timer = 0
        self.is_menu_locked = False 

    def _get_obs_metrics(self):
        sct_img = self.sct.grab(self.monitor)
        raw = np.array(sct_img)[:, :, :3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        
        finish_roi = raw[320:480, 170:470]
        hsv_finish = cv2.cvtColor(finish_roi, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_finish, np.array([0, 150, 100]), np.array([10, 255, 255]))
        red_count = np.count_nonzero(red_mask)
        
        roi_grass = raw[240:, :, :]
        hsv_grass = cv2.cvtColor(roi_grass, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_grass, np.array([35, 50, 50]), np.array([90, 255, 255]))
        green_count = np.count_nonzero(green_mask)
        
        speed_roi = gray[448:473, 540:620]
        _, thresh = cv2.threshold(speed_roi, 245, 255, cv2.THRESH_BINARY)
        speed = float(np.sum(thresh) // 1020)
        
        obs_resized = cv2.resize(gray, (96, 72), interpolation=cv2.INTER_NEAREST)
        wall_check_area = obs_resized[25:50, 30:65]
        
        return np.expand_dims(obs_resized, axis=-1), speed, red_count, green_count, wall_check_area

    def reset_sequence(self):
        global TOTAL_FINISHES
        self.is_menu_locked = True 
        TOTAL_FINISHES += 1
        all_keys_up()
        print(f"\n[SUCCESS] Finish {TOTAL_FINISHES} reached! Clearing menus...")
        
        time.sleep(1.0)
        for i in range(12):
            ui.write(e.EV_KEY, K_ENTER, 1); ui.syn(); time.sleep(0.05)
            ui.write(e.EV_KEY, K_ENTER, 0); ui.syn(); time.sleep(0.6)
            
        print("[INTERLOCK] Track loading...")
        time.sleep(3.0) 
        self.history.clear()
        self.frames_since_reset = 0
        self.low_speed_timer = 0
        self.is_menu_locked = False 

    def reset(self, seed=None, options=None):
        all_keys_up()
        if not self.is_menu_locked:
            ui.write(e.EV_KEY, K_BACK, 1); ui.syn(); time.sleep(0.1); ui.write(e.EV_KEY, K_BACK, 0); ui.syn()
            time.sleep(1.2)
        self.history.clear()
        self.frames_since_reset = 0
        self.low_speed_timer = 0
        obs, _, _, _, _ = self._get_obs_metrics()
        return obs, {}

    def step(self, action):
        global JUST_UPDATED, TOTAL_RESETS
        # If locked, DO NOT perform any AI actions or key presses
        if self.is_menu_locked:
            return np.zeros((72, 96, 1), dtype=np.uint8), 0.0, False, False, {}

        if JUST_UPDATED:
            JUST_UPDATED = False
            return self.reset()[0], 0.0, True, False, {}

        obs, speed, red_count, green_count, wall_area = self._get_obs_metrics()

        if self.frames_since_reset > MAX_RACE_FRAMES:
            print("[RESET] Timeout")
            TOTAL_RESETS += 1
            return obs, -10.0, True, False, {}

        if red_count > RED_PIXEL_THRESHOLD and self.frames_since_reset > 300:
            self.reset_sequence()
            return obs, 100.0, True, False, {}

        if green_count > GREEN_PIXEL_THRESHOLD and self.frames_since_reset > OFFTRACK_GRACE_PERIOD:
            print(f"[RESET] Grass ({green_count}px)")
            TOTAL_RESETS += 1
            return obs, -20.0, True, False, {}

        if self.frames_since_reset > START_GRACE_PERIOD:
            if speed < MIN_SPEED_THRESHOLD:
                self.low_speed_timer += 1
            else:
                self.low_speed_timer = 0
            
            if self.low_speed_timer > 85:
                print(f"[RESET] Stationary ({speed})")
                TOTAL_RESETS += 1
                return obs, -20.0, True, False, {}

            if len(self.history) == STUCK_WINDOW:
                sim = cv2.matchTemplate(wall_area, self.history[0], cv2.TM_CCOEFF_NORMED)[0][0]
                if sim > STUCK_SIMILARITY_LIMIT:
                    print(f"[RESET] Wall - Similarity: {sim:.4f}")
                    TOTAL_RESETS += 1
                    return obs, -20.0, True, False, {}

        # --- AI CONTROL (Only if not locked) ---
        all_keys_up()
        if action == 1: ui.write(e.EV_KEY, K_UP, 1)
        elif action == 2: [ui.write(e.EV_KEY, K_UP, 1), ui.write(e.EV_KEY, K_LEFT, 1)]
        elif action == 3: [ui.write(e.EV_KEY, K_UP, 1), ui.write(e.EV_KEY, K_RIGHT, 1)]
        elif action == 4: ui.write(e.EV_KEY, K_LEFT, 1)
        elif action == 5: ui.write(e.EV_KEY, K_RIGHT, 1)
        ui.syn()

        self.frames_since_reset += 1
        self.history.append(wall_area)
        reward = float(max(0, speed - 5) * 0.01)
        return obs, reward, False, False, {}

if __name__ == "__main__":
    os.makedirs("./models/checkpoints/", exist_ok=True)
    env = DummyVecEnv([lambda: LinuxTMNFEnv()])
    
    if os.path.exists(save_path + ".zip"):
        print(f"Loading model: {save_path}")
        model = PPO.load(save_path, env=env, learning_rate=1e-4)
    else:
        print("Initializing fresh PPO model...")
        model = PPO("CnnPolicy", env, verbose=1, 
                    learning_rate=3e-4, 
                    n_steps=2048, 
                    batch_size=128, 
                    n_epochs=4)
    
    try:
        model.learn(total_timesteps=1000000, callback=TMNFCallback())
    finally:
        all_keys_up()
        model.save(save_path)