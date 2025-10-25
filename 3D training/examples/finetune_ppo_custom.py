<<<<<<< HEAD
"""Custom PPO fine-tuning that directly uses your BC-trained policy.

This approach:
1. Loads your exact BC policy architecture (DeterministicPolicy)
2. Wraps it in a PPO-compatible actor-critic setup
3. Fine-tunes the BC weights using policy gradient RL

Usage:
    python finetune_ppo_custom.py --bc_policy data/episodes/policy_bc.pt --iterations 1000
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import airsim
import cv2
import time

from policy import DeterministicPolicy


class ActorCritic(nn.Module):
    """
    Actor-Critic network that uses your BC policy as the actor initialization.
    """
    def __init__(self, state_dim, bc_policy_path=None):
        super().__init__()
        
        # Actor: use your BC policy architecture
        if bc_policy_path:
            # Load BC policy weights
            self.actor = DeterministicPolicy(state_dim)
            self.actor.load_state_dict(torch.load(bc_policy_path))
            print(f"✓ Loaded BC policy from {bc_policy_path}")
        else:
            # Random initialization
            self.actor = DeterministicPolicy(state_dim)
        
        # Critic: value function (estimates expected return)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Log standard deviation for stochastic policy (exploration)
        self.log_std = nn.Parameter(torch.zeros(3))  # [steering, throttle, brake]
    
    def forward(self, state):
        """Returns action mean, log_std, and value estimate."""
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, self.log_std, value
    
    def get_action_and_value(self, state, action=None):
        """Sample action from policy or evaluate given action."""
        action_mean, log_std, value = self.forward(state)
        std = torch.exp(log_std)
        
        # Create Gaussian distribution for continuous actions
        dist = Normal(action_mean, std)
        
        if action is None:
            # Sample action during rollout
            action = dist.sample()
            
            # Clip to valid ranges
            action[:, 0] = torch.tanh(action[:, 0])  # steering [-1, 1]
            action[:, 1] = torch.sigmoid(action[:, 1])  # throttle [0, 1]
            action[:, 2] = torch.sigmoid(action[:, 2])  # brake [0, 1]
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value


class RolloutBuffer:
    """Stores experience for PPO updates."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.stack(self.values).squeeze(),
            torch.stack(self.log_probs),
            torch.tensor(self.dones, dtype=torch.float32)
        )
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


def collect_rollout(agent, client, n_steps, goal_position, distance_threshold, 
                    step_penalty, success_bonus, collision_penalty, 
                    camera_names, device):
    """Collect experience by running the current policy."""
    buffer = RolloutBuffer()
    
    # Reset environment
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    prev_distance = None
    
    for step in range(n_steps):
        # Get state
        state = get_state(client, goal_position, camera_names, device)
        
        # Get action from policy
        with torch.no_grad():
            action, log_prob, entropy, value = agent.get_action_and_value(state.unsqueeze(0))
        
        action_np = action.squeeze(0).cpu().numpy()
        
        # Apply action
        controls = airsim.CarControls()
        controls.steering = float(action_np[0])
        controls.throttle = float(action_np[1])
        controls.brake = float(action_np[2])
        client.setCarControls(controls)
        time.sleep(0.05)
        
        # Get reward and check termination
        car_state = client.getCarState()
        position = np.array([
            car_state.kinematics_estimated.position.x_val,
            car_state.kinematics_estimated.position.y_val,
            car_state.kinematics_estimated.position.z_val
        ])
        distance_to_goal = np.linalg.norm(position - goal_position)
        
        # Compute reward
        if prev_distance is not None:
            progress = prev_distance - distance_to_goal
        else:
            progress = 0.0
        prev_distance = distance_to_goal
        
        reward = progress - step_penalty
        done = False
        
        collision_info = client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward -= collision_penalty
            done = True
        elif distance_to_goal <= distance_threshold:
            reward += success_bonus
            done = True
        
        # Store experience
        buffer.add(state, action.squeeze(0), reward, value.squeeze(0), 
                   log_prob.squeeze(0), done)
        
        if done:
            break
    
    return buffer


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
    
    returns = advantages + values
    return advantages, returns


def ppo_update(agent, optimizer, buffer, clip_epsilon=0.2, epochs=4, batch_size=64):
    """PPO policy update."""
    states, actions, rewards, values, old_log_probs, dones = buffer.get()
    
    # Compute advantages
    advantages, returns = compute_gae(rewards, values, dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Multiple epochs of optimization
    for epoch in range(epochs):
        # Mini-batch updates
        indices = torch.randperm(len(states))
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            
            # Evaluate actions with current policy
            _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                batch_states, batch_actions
            )
            
            # Policy loss (clipped surrogate objective)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(
                ratio * batch_advantages,
                clipped_ratio * batch_advantages
            ).mean()
            
            # Value loss
            value_loss = 0.5 * (new_values.squeeze() - batch_returns).pow(2).mean()
            
            # Entropy bonus (encourages exploration)
            entropy_loss = -0.01 * entropy.mean()
            
            # Total loss
            loss = policy_loss + value_loss + entropy_loss
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
    
    return policy_loss.item(), value_loss.item()


def get_state(client, goal_position, camera_names, device):
    """Get current state observation."""
    # Get images
    image_request = [
        airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False)
        for cam in camera_names
    ]
    responses = client.simGetImages(image_request)
    
    images = []
    for resp in responses:
        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(resp.height, resp.width, 3)
        img_resized = cv2.resize(img_rgb, (64, 64))
        images.append(img_resized.flatten())
    
    # Get car state
    car_state = client.getCarState()
    position = np.array([
        car_state.kinematics_estimated.position.x_val,
        car_state.kinematics_estimated.position.y_val,
        car_state.kinematics_estimated.position.z_val
    ])
    
    distance_to_goal = np.linalg.norm(position - goal_position)
    speed = car_state.speed
    
    # Concatenate
    state = np.concatenate(images + [[distance_to_goal], [speed]])
    return torch.tensor(state, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_policy', type=str, required=True,
                        help='Path to BC-trained policy (policy_bc.pt)')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of PPO iterations')
    parser.add_argument('--steps_per_iter', type=int, default=2048,
                        help='Steps to collect per iteration')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--device', default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--save_path', default='policy_ppo_finetuned.pt',
                        help='Where to save fine-tuned policy')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    client = airsim.CarClient()
    client.confirmConnection()
    
    camera_names = ["front_center", "front_left", "front_right", "back_center"]
    goal_position = np.array([0.0, 0.0, -1.0])
    distance_threshold = 2.0
    step_penalty = 0.01
    success_bonus = 10.0
    collision_penalty = 50.0
    
    # Get state dimension
    dummy_state = get_state(client, goal_position, camera_names, device)
    state_dim = dummy_state.shape[0]
    
    # Create agent (loads BC policy)
    print("Creating Actor-Critic agent with BC initialization...")
    agent = ActorCritic(state_dim, bc_policy_path=args.bc_policy).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    
    # Training loop
    print(f"Starting PPO fine-tuning for {args.iterations} iterations...")
    for iteration in range(args.iterations):
        # Collect rollout
        buffer = collect_rollout(
            agent, client, args.steps_per_iter, goal_position, distance_threshold,
            step_penalty, success_bonus, collision_penalty, camera_names, device
        )
        
        # PPO update
        policy_loss, value_loss = ppo_update(agent, optimizer, buffer)
        
        # Logging
        if iteration % 10 == 0:
            print(f"Iter {iteration}: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")
        
        # Save checkpoint
        if iteration % 100 == 0:
            torch.save(agent.actor.state_dict(), f"checkpoint_iter_{iteration}.pt")
    
    # Save final policy
    torch.save(agent.actor.state_dict(), args.save_path)
    print(f"✓ Saved fine-tuned policy to {args.save_path}")
    
    client.enableApiControl(False)


if __name__ == '__main__':
    main()
=======
"""Custom PPO fine-tuning that directly uses your BC-trained policy.

This approach:
1. Loads your exact BC policy architecture (DeterministicPolicy)
2. Wraps it in a PPO-compatible actor-critic setup
3. Fine-tunes the BC weights using policy gradient RL

Usage:
    python finetune_ppo_custom.py --bc_policy data/episodes/policy_bc.pt --iterations 1000
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import airsim
import cv2
import time

from policy import DeterministicPolicy


class ActorCritic(nn.Module):
    """
    Actor-Critic network that uses your BC policy as the actor initialization.
    """
    def __init__(self, state_dim, bc_policy_path=None):
        super().__init__()
        
        # Actor: use your BC policy architecture
        if bc_policy_path:
            # Load BC policy weights
            self.actor = DeterministicPolicy(state_dim)
            self.actor.load_state_dict(torch.load(bc_policy_path))
            print(f"✓ Loaded BC policy from {bc_policy_path}")
        else:
            # Random initialization
            self.actor = DeterministicPolicy(state_dim)
        
        # Critic: value function (estimates expected return)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Log standard deviation for stochastic policy (exploration)
        self.log_std = nn.Parameter(torch.zeros(3))  # [steering, throttle, brake]
    
    def forward(self, state):
        """Returns action mean, log_std, and value estimate."""
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, self.log_std, value
    
    def get_action_and_value(self, state, action=None):
        """Sample action from policy or evaluate given action."""
        action_mean, log_std, value = self.forward(state)
        std = torch.exp(log_std)
        
        # Create Gaussian distribution for continuous actions
        dist = Normal(action_mean, std)
        
        if action is None:
            # Sample action during rollout
            action = dist.sample()
            
            # Clip to valid ranges
            action[:, 0] = torch.tanh(action[:, 0])  # steering [-1, 1]
            action[:, 1] = torch.sigmoid(action[:, 1])  # throttle [0, 1]
            action[:, 2] = torch.sigmoid(action[:, 2])  # brake [0, 1]
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value


class RolloutBuffer:
    """Stores experience for PPO updates."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.stack(self.values).squeeze(),
            torch.stack(self.log_probs),
            torch.tensor(self.dones, dtype=torch.float32)
        )
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()


def collect_rollout(agent, client, n_steps, goal_position, distance_threshold, 
                    step_penalty, success_bonus, collision_penalty, 
                    camera_names, device):
    """Collect experience by running the current policy."""
    buffer = RolloutBuffer()
    
    # Reset environment
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    prev_distance = None
    
    for step in range(n_steps):
        # Get state
        state = get_state(client, goal_position, camera_names, device)
        
        # Get action from policy
        with torch.no_grad():
            action, log_prob, entropy, value = agent.get_action_and_value(state.unsqueeze(0))
        
        action_np = action.squeeze(0).cpu().numpy()
        
        # Apply action
        controls = airsim.CarControls()
        controls.steering = float(action_np[0])
        controls.throttle = float(action_np[1])
        controls.brake = float(action_np[2])
        client.setCarControls(controls)
        time.sleep(0.05)
        
        # Get reward and check termination
        car_state = client.getCarState()
        position = np.array([
            car_state.kinematics_estimated.position.x_val,
            car_state.kinematics_estimated.position.y_val,
            car_state.kinematics_estimated.position.z_val
        ])
        distance_to_goal = np.linalg.norm(position - goal_position)
        
        # Compute reward
        if prev_distance is not None:
            progress = prev_distance - distance_to_goal
        else:
            progress = 0.0
        prev_distance = distance_to_goal
        
        reward = progress - step_penalty
        done = False
        
        collision_info = client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward -= collision_penalty
            done = True
        elif distance_to_goal <= distance_threshold:
            reward += success_bonus
            done = True
        
        # Store experience
        buffer.add(state, action.squeeze(0), reward, value.squeeze(0), 
                   log_prob.squeeze(0), done)
        
        if done:
            break
    
    return buffer


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
    
    returns = advantages + values
    return advantages, returns


def ppo_update(agent, optimizer, buffer, clip_epsilon=0.2, epochs=4, batch_size=64):
    """PPO policy update."""
    states, actions, rewards, values, old_log_probs, dones = buffer.get()
    
    # Compute advantages
    advantages, returns = compute_gae(rewards, values, dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Multiple epochs of optimization
    for epoch in range(epochs):
        # Mini-batch updates
        indices = torch.randperm(len(states))
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]
            
            # Evaluate actions with current policy
            _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                batch_states, batch_actions
            )
            
            # Policy loss (clipped surrogate objective)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(
                ratio * batch_advantages,
                clipped_ratio * batch_advantages
            ).mean()
            
            # Value loss
            value_loss = 0.5 * (new_values.squeeze() - batch_returns).pow(2).mean()
            
            # Entropy bonus (encourages exploration)
            entropy_loss = -0.01 * entropy.mean()
            
            # Total loss
            loss = policy_loss + value_loss + entropy_loss
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
    
    return policy_loss.item(), value_loss.item()


def get_state(client, goal_position, camera_names, device):
    """Get current state observation."""
    # Get images
    image_request = [
        airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False)
        for cam in camera_names
    ]
    responses = client.simGetImages(image_request)
    
    images = []
    for resp in responses:
        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(resp.height, resp.width, 3)
        img_resized = cv2.resize(img_rgb, (64, 64))
        images.append(img_resized.flatten())
    
    # Get car state
    car_state = client.getCarState()
    position = np.array([
        car_state.kinematics_estimated.position.x_val,
        car_state.kinematics_estimated.position.y_val,
        car_state.kinematics_estimated.position.z_val
    ])
    
    distance_to_goal = np.linalg.norm(position - goal_position)
    speed = car_state.speed
    
    # Concatenate
    state = np.concatenate(images + [[distance_to_goal], [speed]])
    return torch.tensor(state, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_policy', type=str, required=True,
                        help='Path to BC-trained policy (policy_bc.pt)')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of PPO iterations')
    parser.add_argument('--steps_per_iter', type=int, default=2048,
                        help='Steps to collect per iteration')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--device', default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--save_path', default='policy_ppo_finetuned.pt',
                        help='Where to save fine-tuned policy')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    client = airsim.CarClient()
    client.confirmConnection()
    
    camera_names = ["front_center", "front_left", "front_right", "back_center"]
    goal_position = np.array([0.0, 0.0, -1.0])
    distance_threshold = 2.0
    step_penalty = 0.01
    success_bonus = 10.0
    collision_penalty = 50.0
    
    # Get state dimension
    dummy_state = get_state(client, goal_position, camera_names, device)
    state_dim = dummy_state.shape[0]
    
    # Create agent (loads BC policy)
    print("Creating Actor-Critic agent with BC initialization...")
    agent = ActorCritic(state_dim, bc_policy_path=args.bc_policy).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    
    # Training loop
    print(f"Starting PPO fine-tuning for {args.iterations} iterations...")
    for iteration in range(args.iterations):
        # Collect rollout
        buffer = collect_rollout(
            agent, client, args.steps_per_iter, goal_position, distance_threshold,
            step_penalty, success_bonus, collision_penalty, camera_names, device
        )
        
        # PPO update
        policy_loss, value_loss = ppo_update(agent, optimizer, buffer)
        
        # Logging
        if iteration % 10 == 0:
            print(f"Iter {iteration}: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")
        
        # Save checkpoint
        if iteration % 100 == 0:
            torch.save(agent.actor.state_dict(), f"checkpoint_iter_{iteration}.pt")
    
    # Save final policy
    torch.save(agent.actor.state_dict(), args.save_path)
    print(f"✓ Saved fine-tuned policy to {args.save_path}")
    
    client.enableApiControl(False)


if __name__ == '__main__':
    main()
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
