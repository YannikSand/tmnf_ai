import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import config
from environment import LinuxTMNFEnv
from utils import TMNFCallback, all_keys_up

def main():
    torch.set_num_threads(config.NUM_THREADS)
    os.makedirs("./models/checkpoints/", exist_ok=True)

    env = DummyVecEnv([lambda: LinuxTMNFEnv()])
    
    # Define these clearly for overriding
    params = {
        "n_steps": 8192,
        "batch_size": 512,
        "n_epochs": 10,
        "learning_rate": 2e-4
    }

    if os.path.exists(config.SAVE_PATH + ".zip"):
        print(f"Loading and OVERRIDING model: {config.SAVE_PATH}")
        # Passing params into custom_objects forces the loaded model to use them
        model = PPO.load(config.SAVE_PATH, env=env, custom_objects=params)
    else:
        print("Starting fresh 3-action steering model...")
        model = PPO("CnnPolicy", env, verbose=1, **params)
    
    try:
        model.learn(total_timesteps=1000000, callback=TMNFCallback())
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        all_keys_up()
        model.save(config.SAVE_PATH)
        print("Saved.")

if __name__ == "__main__":
    main()