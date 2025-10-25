"""Load a trained SB3 DQN policy and run deterministic test episodes in AirSim.

Usage:
    python test_dqn.py --model dqn_airsim_car_policy --n_episodes 5

This assumes the model was trained with the same observation preprocessing (HxWxC -> CxHxW).
The script instantiates `AirSimCarEnv`, wraps it to transpose observations to channel-first,
loads the DQN model, and runs episodes while printing simple statistics.
"""
import argparse
import time
import numpy as np

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from airgym.envs.car_env import AirSimCarEnv
from gymnasium import ObservationWrapper, spaces


class TransposeImageWrapper(ObservationWrapper):
    """Transpose HxWxC image obs to CxHxW for PyTorch/CNN policies.

    Accepts grayscale HxW or HxWx1 inputs as well.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) not in (2, 3):
            raise ValueError("TransposeImageWrapper expects a Box observation space with shape (H,W) or (H,W,C)")
        if len(obs_space.shape) == 2:
            h, w = obs_space.shape
            c = 1
        else:
            h, w, c = obs_space.shape
        low = float(obs_space.low.min())
        high = float(obs_space.high.max())
        self.observation_space = spaces.Box(low=low, high=high, shape=(c, h, w), dtype=obs_space.dtype)

    def observation(self, obs):
        if obs.ndim == 2:
            obs = np.expand_dims(obs, axis=2)
        return np.transpose(obs, (2, 0, 1))


def make_env_instance(ip_address="127.0.0.1", image_shape=(84, 84, 1)):
    env = AirSimCarEnv(ip_address=ip_address, image_shape=image_shape)
    env = TransposeImageWrapper(env)
    env = Monitor(env)
    return env


def test_model(model_path: str, n_episodes: int, ip_address: str):
    print(f"Loading model from: {model_path}")
    model = DQN.load(model_path)

    env = make_env_instance(ip_address=ip_address)

    # If model was not saved with env, attach it now
    try:
        model.set_env(env)
    except Exception:
        # Not critical; model.predict works without env set
        pass

    results = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        start = time.time()
        done = False
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            # For discrete action_space, action may be array-like
            if isinstance(action, (np.ndarray, list, tuple)):
                action = int(np.asarray(action).squeeze())
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1
            if steps % 50 == 0:
                print(f"  Ep {ep+1} step {steps}: reward_sum={total_reward:.2f}")
        duration = time.time() - start
        print(f"Episode {ep+1}/{n_episodes} finished: steps={steps}, total_reward={total_reward:.2f}, time={duration:.1f}s")
        results.append({"steps": steps, "reward": total_reward})

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dqn_airsim_car_policy",
                        help="Path to SB3 model (without .zip) or full path")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="AirSim IP address")
    args = parser.parse_args()

    results = test_model(args.model, args.n_episodes, args.ip)
    avg_reward = np.mean([r["reward"] for r in results]) if results else 0.0
    avg_steps = int(np.mean([r["steps"] for r in results])) if results else 0
    print(f"\nSummary: avg_reward={avg_reward:.2f}, avg_steps={avg_steps}")


if __name__ == "__main__":
    main()
