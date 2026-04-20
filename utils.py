import time
from evdev import UInput, ecodes as e
from stable_baselines3.common.callbacks import BaseCallback
from pynput import keyboard
import config

ui = UInput()

def all_keys_up():
    for key_code in config.KEYS.values():
        ui.write(e.EV_KEY, key_code, 0)
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
        # Pause the brain if the environment is clearing menus
        if self.training_env.get_attr("is_menu_locked")[0]:
            return True 

        self.step_counter += 1
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            fps = self.step_counter / max(elapsed, 1e-3)
            finishes = self.training_env.get_attr("total_finishes")[0]
            print(f">>> [STATS] Step: {self.num_timesteps} | Drive FPS: {fps:.2f} | Finishes: {finishes}")
            self.start_time = time.time()
            self.step_counter = 0
        return not self.stop_training

    def _on_rollout_end(self):
        all_keys_up()
        # Signals the environment that a brain update just finished
        self.training_env.env_method("signal_update_finished")