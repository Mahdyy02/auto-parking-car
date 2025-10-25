import gymnasium as gym
import airgym  # ensure local env registration (airgym.__init__ registers env ids)
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# Create a DummyVecEnv for main airsim gym env
from airgym.envs.car_env import AirSimCarEnv
import numpy as np
from gymnasium import ObservationWrapper, spaces


class TransposeImageWrapper(ObservationWrapper):
    """Transpose HxWxC image obs to CxHxW for PyTorch/CNN policies.

    This is a small local alternative to `VecTransposeImage` applied per-env.
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
        low = obs_space.low.min()
        high = obs_space.high.max()
        self.observation_space = spaces.Box(low=low, high=high, shape=(c, h, w), dtype=obs_space.dtype)

    def observation(self, obs):
        # If grayscale HxW, add channel dim
        if obs.ndim == 2:
            obs = np.expand_dims(obs, axis=2)
        return np.transpose(obs, (2, 0, 1))

def make_env():
    # instantiate the env class directly to avoid registry/name issues
    env = AirSimCarEnv(ip_address="127.0.0.1", image_shape=(84, 84, 1))
    env = TransposeImageWrapper(env)
    return Monitor(env)

env = DummyVecEnv([make_env])

# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=200000,
    buffer_size=5000,
    max_grad_norm=10,
    exploration_fraction=0.01,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5, tb_log_name="dqn_airsim_car_run_" + str(time.time()), **kwargs
)

# Save policy weights
model.save("dqn_airsim_car_policy")
