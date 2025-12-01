# RLModel.py
import os
import time
import gymnasium as gym
import numpy as np
import torch

# Import your environment file (this will register 'Tetris-v0')
import Tetris  # assumes Tetris.py registers 'Tetris-v0'

# Stable Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# ---------------- Config ----------------
# You can tweak these values
NUM_ENVS = 24         # number of parallel envs used for training
TOTAL_TIMESTEPS = 200_000
CHECKPOINT_FREQ = 50_000
SAVE_DIR = "models"
LOG_DIR = "logs/tensorboard"
# Architectures to try (each will be trained sequentially)
ARCHITECTURES = [
    
    [32, 32, 32],
]
# Training hyperparams
LEARNING_RATE = 1e-4
BATCH_SIZE = 512    # larger batch helps utilize GPU
BUFFER_SIZE = 100_000
TRAIN_FREQ = 4      # steps between learning updates
GRADIENT_STEPS = 1
VERBOSE = 1

# Render settings for demo after training
RENDER_EPISODES = 3
RENDER_MAX_STEPS = 5000
RENDER_EVERY_N = 3   # only render every N environment steps during demo (speeds things up)
RENDER_SLEEP = 0.02  # seconds sleep between rendered frames (controls play speed)

# -------------- Helper functions --------------
def make_env_fn():
    """
    Return a function that creates and returns a Monitor-wrapped Tetris env.
    Using a top-level function is necessary for subprocess pickling.
    """
    def _init():
        env = gym.make("Tetris-v0")
        env = Monitor(env)
        return env
    return _init

def render_demo(model, episodes=RENDER_EPISODES, max_steps=RENDER_MAX_STEPS, render_every=RENDER_EVERY_N, sleep=RENDER_SLEEP):
    """
    Run a few deterministic episodes with rendering so you can watch the agent play.
    Renders only every `render_every` steps to speed up playback while still showing gameplay.
    """
    env = gym.make("Tetris-v0")  # non-vectorized for clear rendering
    try:
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            step = 0
            while not done and step < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

                if (step % render_every) == 0:
                    # safe render with event handling: env.render() should be non-blocking
                    try:
                        env.render()
                    except Exception:
                        # in case render crashes, ignore and continue
                        pass
                    # small sleep so the render can be visually followed
                    time.sleep(sleep)
                step += 1
            print(f"[demo] episode {ep+1} finished in {step} steps.")
    finally:
        try:
            env.close()
        except Exception:
            pass

# --------------- Main ---------------
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Device (GPU) selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if device == "cuda":
        try:
            print("CUDA device count:", torch.cuda.device_count())
            print("CUDA device name:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("CUDA info error:", e)

    # Quick env check on a single env before parallelizing
    print("Creating single env for checks...")
    single_env = Monitor(gym.make("Tetris-v0"))
    print("Running stable-baselines env check (warn=True)...")
    check_env(single_env, warn=True)
    try:
        single_env.close()
    except Exception:
        pass

    # Choose vectorized env: SubprocVecEnv (parallel) to feed GPU more data
    print(f"Creating SubprocVecEnv with {NUM_ENVS} envs...")
    env_fns = [make_env_fn() for _ in range(NUM_ENVS)]

    venv = SubprocVecEnv(env_fns)

    # Train sequentially for each architecture in ARCHITECTURES
    for arch_idx, net_arch in enumerate(ARCHITECTURES):
        print("\n" + "="*70)
        print(f"Training architecture {arch_idx}: net_arch = {net_arch}")
        print("="*70)

        policy_kwargs = dict(net_arch=net_arch)

        model = DQN(
            policy="MultiInputPolicy",
            env=venv,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=2_000,
            batch_size=BATCH_SIZE,
            tau=1.0,
            gamma=0.99,
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS,
            verbose=VERBOSE,
            tensorboard_log=os.path.join(LOG_DIR, f"arch_{arch_idx}"),
            policy_kwargs=policy_kwargs,
            device=device,
        )

        checkpoint_cb = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=os.path.join(SAVE_DIR, f"arch_{arch_idx}"), name_prefix=f"dqn_arch{arch_idx}")

        # Train (we reduce logging frequency / heavy logging)
        t0 = time.time()
        print("Starting model.learn() ... (no rendering during training)")
        # set log_interval=1 to avoid frequent TensorBoard overhead; you can increase if you want logs.
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb, log_interval=10)
        elapsed = time.time() - t0
        print(f"Training finished for arch {arch_idx} in {elapsed:.1f} seconds.")

        # Save final model
        model_path = os.path.join(SAVE_DIR, f"dqn_tetris_arch{arch_idx}")
        model.save(model_path)
        print("Saved model to:", model_path)

        # Print model summary (policy + q_net if available)
        print("\nModel policy summary:")
        print(model.policy)
        try:
            print("\nModel q_net:")
            # model.q_net exists for DQN in some versions; otherwise use model.policy.q_net
            if hasattr(model, "q_net"):
                print(model.q_net)
            else:
                print(model.policy.q_net)
        except Exception as e:
            print("Could not print q_net:", e)

        # Close the SubprocVecEnv between runs and recreate fresh envs to avoid stale processes
        try:
            venv.close()
        except Exception:
            pass

        # Run rendered demo (separate process / after training)
        print("\nRunning rendered demo (after training)...")
        try:
            render_demo(model, episodes=RENDER_EPISODES, max_steps=RENDER_MAX_STEPS, render_every=RENDER_EVERY_N, sleep=RENDER_SLEEP)
        except Exception as e:
            print("Error during render_demo():", e)

        # Recreate venv for next architecture (if any)
        print("Recreating vectorized env for next architecture...")
        venv = SubprocVecEnv([make_env_fn() for _ in range(NUM_ENVS)])

    # Final cleanup
    try:
        venv.close()
    except Exception:
        pass

    print("All architectures completed. Models saved in", SAVE_DIR)
