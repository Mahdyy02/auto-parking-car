"""Evaluate a trained policy by running it in AirSim simulator.

This script:
1. Loads your trained BC policy (policy_bc.pt)
2. Runs episodes in the simulator using the policy
3. Shows real-time metrics and success rate
4. Optionally saves video/visualization

Usage:
    # Evaluate BC policy
    python eval_policy.py --policy data/episodes/policy_bc.pt --n_episodes 10
    
    # With visualization
    python eval_policy.py --policy data/episodes/policy_bc.pt --n_episodes 5 --render
"""
import os
import argparse
import time
from typing import Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import airsim


class DeterministicPolicy(nn.Module):
    """Simple MLP policy for behavioral cloning - matches record_manual_policy.py architecture."""
    def __init__(self, state_dim, action_dim=3, hidden_dims=[1024, 512, 256]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)
        
    def forward(self, state):
        features = self.backbone(state)
        actions = self.action_head(features)
        
        # Output: [steering, throttle, brake]
        # steering in [-1, 1], throttle in [-1, 1], brake in [0, 1]
        steering = torch.tanh(actions[:, 0:1])
        throttle = torch.tanh(actions[:, 1:2])
        brake = torch.sigmoid(actions[:, 2:3])
        
        return torch.cat([steering, throttle, brake], dim=1)


def image_embedding_placeholder(img: np.ndarray, size=(64, 64)) -> np.ndarray:
    """Same embedding as used during training."""
    im = cv2.resize(img, size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    return im.ravel()


def get_images(client: airsim.CarClient, camera_names, vehicle_name='Car1') -> Tuple[np.ndarray, ...]:
    """Get images from cameras."""
    reqs = [airsim.ImageRequest(name, airsim.ImageType.Scene, False, False) for name in camera_names]
    responses = client.simGetImages(reqs, vehicle_name=vehicle_name)
    imgs = []
    for r in responses:
        if r.height == 0:
            imgs.append(np.zeros((16, 16, 3), dtype=np.uint8))
            continue
        arr = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
        img = arr.reshape((r.height, r.width, 3))
        imgs.append(img)
    return tuple(imgs)


def get_state(client, camera_names, goal_pos, vehicle_name='Car1'):
    """Get current state observation (same as training)."""
    # Get images
    imgs = get_images(client, camera_names, vehicle_name)
    
    # Embed images
    embeddings = [image_embedding_placeholder(im, size=(64, 64)) for im in imgs]
    img_emb = np.concatenate(embeddings)
    
    # Get car state
    cs = client.getCarState(vehicle_name=vehicle_name)
    pos = cs.kinematics_estimated.position
    car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
    
    vel = cs.kinematics_estimated.linear_velocity
    car_speed = float(np.linalg.norm([vel.x_val, vel.y_val, vel.z_val]))
    
    # Distance to goal
    dist_to_goal = float(np.linalg.norm(car_pos - goal_pos))
    
    # Distance to obstacle (placeholder - you can add geometric calculation here)
    dist_to_obstacle = 10.0  # Placeholder
    
    # Concatenate state
    state = np.concatenate([img_emb, [dist_to_obstacle], [dist_to_goal], [car_speed]])
    
    return state, car_pos, dist_to_goal, car_speed


def run_episode(client, policy, args, episode_num=0):
    """Run one episode using the policy."""
    camera_names = args.cameras
    goal_pos = np.array([args.goal_x, args.goal_y, args.goal_z], dtype=float)
    goal_threshold = args.goal_threshold
    max_steps = args.max_steps
    
    # Reset (optional)
    if args.reset_before_episode:
        client.reset()
        time.sleep(0.5)
    
    # Enable API control
    client.enableApiControl(True, vehicle_name='Car1')
    try:
        client.armDisarm(True, vehicle_name='Car1')
    except:
        pass
    
    # Release handbrake at start
    initial_controls = airsim.CarControls()
    initial_controls.handbrake = False
    client.setCarControls(initial_controls, vehicle_name='Car1')
    time.sleep(0.1)
    
    print(f"\n{'='*80}")
    print(f"Episode {episode_num + 1}/{args.n_episodes}")
    print(f"{'='*80}")
    
    # Get initial position
    initial_state, initial_pos, initial_dist, _ = get_state(client, camera_names, goal_pos)
    print(f"Starting position: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f}, {initial_pos[2]:.2f})")
    print(f"Goal position:     ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})")
    print(f"Initial distance:  {initial_dist:.2f} m")
    print()
    
    total_reward = 0.0
    prev_dist = initial_dist
    step = 0
    done = False
    
    trajectory_positions = [initial_pos.copy()]
    
    while not done and step < max_steps:
        step += 1
        
        # Get current state
        state, car_pos, dist_to_goal, car_speed = get_state(client, camera_names, goal_pos)
        
        # Policy prediction
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action = policy(state_tensor).squeeze(0).cpu().numpy()
        
        steering, throttle, brake = action[0], action[1], action[2]
        
        # DEBUG: Override with fixed controls to test if car moves
        throttle = 1.0  # Full throttle forward
        brake = 0.0     # No brake
        steering = 0.0  # Straight ahead
        
        # Apply action
        controls = airsim.CarControls()
        controls.steering = float(steering)
        controls.throttle = float(throttle)
        controls.brake = float(brake)
        controls.handbrake = False  # Make sure handbrake is released!
        
        # Gear shifting (critical for car to move!)
        controls.is_manual_gear = True
        if throttle < 0:
            controls.manual_gear = -1  # Reverse
        elif brake > 0.5 or (abs(throttle) < 0.1 and brake > 0.1):
            controls.manual_gear = 0  # Neutral (only if significant braking or coasting)
        else:
            # Forward gears based on speed
            if car_speed < 5:
                controls.manual_gear = 1
            elif car_speed < 10:
                controls.manual_gear = 2
            elif car_speed < 15:
                controls.manual_gear = 3
            else:
                controls.manual_gear = 4
        
        client.setCarControls(controls, vehicle_name='Car1')
        
        # Wait for physics
        time.sleep(args.step_time)
        
        # Calculate reward (same as training)
        progress = prev_dist - dist_to_goal
        reward = progress - args.step_penalty
        prev_dist = dist_to_goal
        
        # Check termination
        collision_info = client.simGetCollisionInfo(vehicle_name='Car1')
        has_collided = collision_info.has_collided
        
        if has_collided:
            reward -= args.collision_penalty
            done = True
            outcome = "COLLISION ✗"
            print(f"  Step {step}: Collision detected!")
        elif dist_to_goal <= goal_threshold:
            reward += args.success_bonus
            done = True
            outcome = "SUCCESS ✓"
            print(f"  Step {step}: Goal reached!")
        else:
            outcome = "ongoing"
        
        total_reward += reward
        trajectory_positions.append(car_pos.copy())
        
        # Print step info
        if step % 5 == 0 or done:
            print(f"  Step {step:3d}: dist={dist_to_goal:6.2f}m, speed={car_speed:5.2f}, "
                  f"steer={steering:+.3f}, throttle={throttle:.3f}, brake={brake:.3f}, "
                  f"reward={reward:+7.2f}")
        
        if done:
            break
    
    if not done:
        outcome = "TIMEOUT"
        print(f"  Step {step}: Max steps reached")
    
    # Summary
    print()
    print(f"Episode Summary:")
    print(f"  Outcome:       {outcome}")
    print(f"  Steps:         {step}")
    print(f"  Total Reward:  {total_reward:.2f}")
    print(f"  Final Distance: {dist_to_goal:.2f} m")
    print(f"  Distance Traveled: {initial_dist - dist_to_goal:.2f} m")
    
    # Stop the car
    controls = airsim.CarControls()
    controls.throttle = 0
    controls.brake = 1.0
    client.setCarControls(controls, vehicle_name='Car1')
    
    return {
        'outcome': outcome,
        'steps': step,
        'total_reward': total_reward,
        'initial_dist': initial_dist,
        'final_dist': dist_to_goal,
        'trajectory': trajectory_positions,
        'success': outcome == "SUCCESS ✓"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to trained policy (.pt file)')
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--cameras', nargs='+', default=['0', 'rear', 'left', 'right'],
                        help='Camera names')
    parser.add_argument('--goal_x', type=float, default=56.488)
    parser.add_argument('--goal_y', type=float, default=3.012)
    parser.add_argument('--goal_z', type=float, default=-0.639)
    parser.add_argument('--goal_threshold', type=float, default=1.0)
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--step_time', type=float, default=0.2,
                        help='Time between steps (seconds)')
    parser.add_argument('--step_penalty', type=float, default=0.01)
    parser.add_argument('--success_bonus', type=float, default=10.0)
    parser.add_argument('--collision_penalty', type=float, default=50.0)
    parser.add_argument('--reset_before_episode', action='store_true',
                        help='Reset simulator before each episode')
    parser.add_argument('--render', action='store_true',
                        help='Show visualization window')
    parser.add_argument('--device', default='cpu',
                        help='Device (cpu or cuda)')
    args = parser.parse_args()
    
    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.CarClient()
    client.confirmConnection()
    print("✓ Connected")
    
    # Load policy
    print(f"\nLoading policy from {args.policy}...")
    
    # Load checkpoint
    checkpoint = torch.load(args.policy, map_location=args.device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
        # New format from record_manual_policy.py
        state_dim = checkpoint['state_dim']
        policy = DeterministicPolicy(state_dim)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"✓ Loaded policy (new format, state_dim={state_dim})")
    elif isinstance(checkpoint, dict) and 'net.0.weight' in checkpoint:
        # Old format - direct state dict
        dummy_img_emb = 4 * 64 * 64 * 3  # 4 cameras × 64×64×3
        dummy_scalars = 3  # obstacle_dist, goal_dist, speed
        state_dim = dummy_img_emb + dummy_scalars
        policy = DeterministicPolicy(state_dim)
        policy.load_state_dict(checkpoint)
        print(f"✓ Loaded policy (old format, state_dim={state_dim})")
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'not a dict'}")
    
    policy.eval()
    
    # Run evaluation
    print(f"\nRunning {args.n_episodes} episodes...")
    print("Watch the AirSim simulator window to see the policy in action!")
    
    results = []
    for i in range(args.n_episodes):
        result = run_episode(client, policy, args, episode_num=i)
        results.append(result)
        
        # Small delay between episodes
        time.sleep(1.0)
    
    # Final statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    successes = sum(1 for r in results if r['success'])
    collisions = sum(1 for r in results if 'COLLISION' in r['outcome'])
    timeouts = sum(1 for r in results if 'TIMEOUT' in r['outcome'])
    
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_dist_traveled = np.mean([r['initial_dist'] - r['final_dist'] for r in results])
    
    print(f"Total Episodes:     {args.n_episodes}")
    print(f"Successes:          {successes} ({100*successes/args.n_episodes:.1f}%)")
    print(f"Collisions:         {collisions} ({100*collisions/args.n_episodes:.1f}%)")
    print(f"Timeouts:           {timeouts} ({100*timeouts/args.n_episodes:.1f}%)")
    print()
    print(f"Average Steps:      {avg_steps:.1f}")
    print(f"Average Reward:     {avg_reward:.2f}")
    print(f"Average Distance:   {avg_dist_traveled:.2f} m")
    
    # Success criteria feedback
    print()
    print("="*80)
    print("PERFORMANCE ASSESSMENT")
    print("="*80)
    
    success_rate = 100 * successes / args.n_episodes
    
    if success_rate >= 80:
        print("🎉 EXCELLENT! Policy performs very well")
        print("   Consider fine-tuning with RL for even better performance")
    elif success_rate >= 60:
        print("✓ GOOD! Policy learned meaningful behavior")
        print("   You can either:")
        print("   - Collect more diverse demonstrations")
        print("   - Fine-tune with RL (PPO)")
    elif success_rate >= 40:
        print("⚠️  MODERATE: Policy learned some behavior but needs improvement")
        print("   Recommendations:")
        print("   - Collect 2-3x more demonstrations")
        print("   - Ensure demonstrations are diverse (different trajectories)")
        print("   - Check if demonstrations have good steering variance")
    elif success_rate >= 20:
        print("❌ POOR: Policy learned minimal behavior")
        print("   Problems likely:")
        print("   - Training data quality is low")
        print("   - Task setup is incorrect (check distances)")
        print("   - Not enough diverse demonstrations")
    else:
        print("❌ FAILED: Policy didn't learn anything useful")
        print("   Critical issues:")
        print("   - Training data is probably bad (check with inspect_episode.py)")
        print("   - Task might be too easy/trivial (episodes too short)")
        print("   - Need to recollect with proper manual control")
    
    # Disable API control
    client.enableApiControl(False, vehicle_name='Car1')
    print("\n✓ Evaluation complete")


if __name__ == '__main__':
    main()
