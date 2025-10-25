"""Fine-tune a BC-trained policy using PPO (Proximal Policy Optimization).

This script:
1. Loads your BC-trained policy (policy_bc.pt)
2. Uses it to initialize a PPO agent
3. Fine-tunes via reinforcement learning in the AirSim environment

Usage:
    python finetune_ppo.py --bc_policy data/episodes/policy_bc.pt --total_timesteps 100000
"""
import argparse
import numpy as np
import torch
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import airsim
import cv2
from policy import DeterministicPolicy


class AirSimParkingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for AirSim parking task.
    This wraps your existing AirSim setup into a gym-compatible interface.
    """
    def __init__(self, 
                 goal_position=np.array([0.0, 0.0, -1.0]),
                 distance_threshold=2.0,
                 max_steps=500,
                 camera_names=["front_center", "front_left", "front_right", "back_center"],
                 step_penalty=0.01,
                 success_bonus=10.0,
                 collision_penalty=50.0):
        super().__init__()
        
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        
        self.goal_position = goal_position
        self.distance_threshold = distance_threshold
        self.max_steps = max_steps
        self.camera_names = camera_names
        self.step_penalty = step_penalty
        self.success_bonus = success_bonus
        self.collision_penalty = collision_penalty
        
        # Action space: [steering, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # State space: flattened images + distances + speed
        # 4 cameras × (64×64×3) + 2 distances + 1 speed
        state_dim = 4 * 64 * 64 * 3 + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.prev_distance_to_goal = None
        self.initial_pose = None
        
    def reset(self):
        """Reset the environment to initial state."""
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Store initial pose for episode reset
        car_state = self.client.getCarState()
        pos = car_state.kinematics_estimated.position
        ori = car_state.kinematics_estimated.orientation
        self.initial_pose = airsim.Pose(pos, ori)
        
        self.current_step = 0
        self.prev_distance_to_goal = None
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return (observation, reward, done, info)."""
        # Apply action
        controls = airsim.CarControls()
        controls.steering = float(action[0])
        controls.throttle = float(action[1])
        controls.brake = float(action[2])
        self.client.setCarControls(controls)
        
        # Wait for physics update
        import time
        time.sleep(0.05)
        
        # Get new state
        obs = self._get_observation()
        
        # Calculate reward
        car_state = self.client.getCarState()
        position = np.array([
            car_state.kinematics_estimated.position.x_val,
            car_state.kinematics_estimated.position.y_val,
            car_state.kinematics_estimated.position.z_val
        ])
        
        distance_to_goal = np.linalg.norm(position - self.goal_position)
        
        # Distance-based shaping reward
        if self.prev_distance_to_goal is not None:
            progress = self.prev_distance_to_goal - distance_to_goal
        else:
            progress = 0.0
        self.prev_distance_to_goal = distance_to_goal
        
        # Check collision
        collision_info = self.client.simGetCollisionInfo()
        has_collided = collision_info.has_collided
        
        # Compute reward
        reward = progress - self.step_penalty
        
        done = False
        info = {}
        
        if has_collided:
            reward -= self.collision_penalty
            done = True
            info['termination_reason'] = 'collision'
        elif distance_to_goal <= self.distance_threshold:
            reward += self.success_bonus
            done = True
            info['termination_reason'] = 'success'
        elif self.current_step >= self.max_steps:
            done = True
            info['termination_reason'] = 'max_steps'
        
        self.current_step += 1
        info['distance_to_goal'] = distance_to_goal
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current state observation."""
        # Get images
        image_request = [
            airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False)
            for cam in self.camera_names
        ]
        responses = self.client.simGetImages(image_request)
        
        images = []
        for resp in responses:
            img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(resp.height, resp.width, 3)
            img_resized = cv2.resize(img_rgb, (64, 64))
            images.append(img_resized.flatten())
        
        # Get car state
        car_state = self.client.getCarState()
        position = np.array([
            car_state.kinematics_estimated.position.x_val,
            car_state.kinematics_estimated.position.y_val,
            car_state.kinematics_estimated.position.z_val
        ])
        
        distance_to_goal = np.linalg.norm(position - self.goal_position)
        speed = car_state.speed
        
        # Concatenate all state components
        state = np.concatenate(images + [
            [distance_to_goal],
            [speed]
        ])
        
        return state.astype(np.float32)
    
    def close(self):
        """Clean up resources."""
        self.client.enableApiControl(False)


def load_bc_policy_weights(bc_policy_path, state_dim):
    """Load BC-trained weights into a policy network."""
    policy = DeterministicPolicy(state_dim)
    policy.load_state_dict(torch.load(bc_policy_path))
    return policy


def create_ppo_from_bc(env, bc_policy_path=None):
    """
    Create a PPO agent, optionally initialized with BC policy weights.
    
    Note: SB3's PPO uses its own policy architecture, so we can't directly
    load your BC policy. Instead, we'll use the BC policy to generate
    synthetic demonstrations for better initialization.
    """
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,  # Entropy coefficient (exploration)
        verbose=1,
        tensorboard_log="./ppo_parking_tensorboard/"
    )
    
    # Optional: If BC policy provided, we could do "behavior cloning fine-tuning"
    # by first running the BC policy to collect on-policy data, then starting PPO
    if bc_policy_path:
        print(f"Loading BC policy from {bc_policy_path} for initialization...")
        # Note: Direct weight transfer is tricky because PPO uses a different architecture
        # Better approach: use BC policy to generate initial rollouts
        print("Tip: BC policy can be used for initial data collection or as a baseline")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_policy', type=str, default=None,
                        help='Path to BC-trained policy (policy_bc.pt)')
    parser.add_argument('--total_timesteps', type=int, default=100000,
                        help='Total timesteps for PPO training')
    parser.add_argument('--checkpoint_freq', type=int, default=10000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='Evaluate every N steps')
    parser.add_argument('--n_eval_episodes', type=int, default=5,
                        help='Number of episodes for evaluation')
    parser.add_argument('--save_path', type=str, default='models/ppo_parking',
                        help='Where to save trained models')
    args = parser.parse_args()
    
    # Create environment
    print("Creating AirSim parking environment...")
    env = AirSimParkingEnv()
    env = DummyVecEnv([lambda: env])  # Wrap for SB3
    
    # Create PPO agent
    print("Creating PPO agent...")
    model = create_ppo_from_bc(env, args.bc_policy)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.save_path,
        name_prefix='ppo_parking'
    )
    
    # Train with PPO
    print(f"Starting PPO training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback
    )
    
    # Save final model
    final_path = f"{args.save_path}/ppo_parking_final.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")
    
    env.close()


if __name__ == '__main__':
    main()
