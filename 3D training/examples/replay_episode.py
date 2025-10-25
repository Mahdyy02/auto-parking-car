"""Replay a recorded episode to verify data collection quality.

This script:
1. Loads a recorded episode (.npz file)
2. Replays the EXACT actions that were recorded
3. Compares the actual trajectory with what was recorded
4. Helps diagnose if recording/replay is working correctly

Usage:
    # Replay a specific episode
    python replay_episode.py --episode data/episodes/episode_1760374918_c96c401b.npz
    
    # Replay a random episode
    python replay_episode.py --random
"""
import os
import argparse
import glob
import random
import time
import numpy as np
import airsim


def replay_episode(episode_path, step_time=0.2, reset_first=True):
    """Replay an episode by executing the recorded actions."""
    
    print("="*80)
    print(f"REPLAYING EPISODE: {os.path.basename(episode_path)}")
    print("="*80)
    
    # Load episode data
    print("\nLoading episode data...")
    data = np.load(episode_path)
    
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    dones = data['dones']
    
    n_steps = len(actions)
    total_recorded_reward = rewards.sum()
    
    print(f"✓ Loaded episode")
    print(f"  Steps: {n_steps}")
    print(f"  Total recorded reward: {total_recorded_reward:.2f}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  States shape: {states.shape}")
    print()
    
    # Analyze recorded actions
    print("Recorded Actions Summary:")
    print(f"  Steering:  min={actions[:, 0].min():+.3f}, max={actions[:, 0].max():+.3f}, "
          f"mean={actions[:, 0].mean():+.3f}, std={actions[:, 0].std():.3f}")
    print(f"  Throttle:  min={actions[:, 1].min():+.3f}, max={actions[:, 1].max():+.3f}, "
          f"mean={actions[:, 1].mean():+.3f}, std={actions[:, 1].std():.3f}")
    print(f"  Brake:     min={actions[:, 2].min():+.3f}, max={actions[:, 2].max():+.3f}, "
          f"mean={actions[:, 2].mean():+.3f}, std={actions[:, 2].std():.3f}")
    print()
    
    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.CarClient()
    client.confirmConnection()
    print("✓ Connected")
    
    # Reset if requested
    if reset_first:
        print("\nResetting simulator...")
        client.reset()
        time.sleep(1.0)
        print("✓ Reset complete")
    
    # Enable API control
    client.enableApiControl(True, vehicle_name='Car1')
    try:
        client.armDisarm(True, vehicle_name='Car1')
    except:
        pass
    
    # Get initial position
    cs = client.getCarState(vehicle_name='Car1')
    initial_pos = cs.kinematics_estimated.position
    print(f"\nStarting position: ({initial_pos.x_val:.2f}, {initial_pos.y_val:.2f}, {initial_pos.z_val:.2f})")
    
    print("\n" + "="*80)
    print("REPLAYING ACTIONS - Watch the simulator!")
    print("="*80)
    print("Press Ctrl+C to stop\n")
    
    # Replay actions
    try:
        for step in range(n_steps):
            steering, throttle, brake = actions[step]
            
            # Apply the recorded action
            controls = airsim.CarControls()
            controls.steering = float(steering)
            controls.throttle = float(throttle)
            controls.brake = float(brake)
            client.setCarControls(controls, vehicle_name='Car1')
            
            # Get current state
            cs = client.getCarState(vehicle_name='Car1')
            pos = cs.kinematics_estimated.position
            vel = cs.kinematics_estimated.linear_velocity
            speed = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])
            
            # Check collision
            collision_info = client.simGetCollisionInfo(vehicle_name='Car1')
            has_collided = collision_info.has_collided
            
            # Print step info
            collision_marker = " [COLLISION!]" if has_collided else ""
            print(f"Step {step+1:3d}/{n_steps}: "
                  f"steer={steering:+.3f}, throttle={throttle:.3f}, brake={brake:.3f} | "
                  f"pos=({pos.x_val:6.2f}, {pos.y_val:6.2f}, {pos.z_val:6.2f}) | "
                  f"speed={speed:5.2f}{collision_marker}")
            
            # Wait for physics update
            time.sleep(step_time)
            
            # Stop if collision
            if has_collided:
                print("\n⚠️  Collision detected during replay!")
                break
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Replay stopped by user")
    
    # Stop the car
    print("\nStopping car...")
    controls = airsim.CarControls()
    controls.throttle = 0
    controls.brake = 1.0
    client.setCarControls(controls, vehicle_name='Car1')
    
    # Get final position
    cs = client.getCarState(vehicle_name='Car1')
    final_pos = cs.kinematics_estimated.position
    
    print(f"\nFinal position: ({final_pos.x_val:.2f}, {final_pos.y_val:.2f}, {final_pos.z_val:.2f})")
    
    # Calculate distance traveled
    distance_traveled = np.sqrt(
        (final_pos.x_val - initial_pos.x_val)**2 +
        (final_pos.y_val - initial_pos.y_val)**2 +
        (final_pos.z_val - initial_pos.z_val)**2
    )
    
    print(f"Distance traveled: {distance_traveled:.2f} m")
    
    print("\n" + "="*80)
    print("REPLAY COMPLETE")
    print("="*80)
    print("\nObservations:")
    print("1. Did the car move as expected?")
    print("2. Was the trajectory smooth or jerky?")
    print("3. Did it look like intentional driving or random movement?")
    print("4. If steering was mostly 0.0, did the car go straight?")
    
    # Disable API control
    client.enableApiControl(False, vehicle_name='Car1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=str, default=None,
                        help='Path to specific episode file to replay')
    parser.add_argument('--random', action='store_true',
                        help='Replay a random episode from data/episodes')
    parser.add_argument('--data_dir', type=str, default='data/episodes',
                        help='Directory containing episodes (used with --random)')
    parser.add_argument('--step_time', type=float, default=0.2,
                        help='Time between steps (seconds)')
    parser.add_argument('--no_reset', action='store_true',
                        help='Do not reset simulator before replay')
    args = parser.parse_args()
    
    # Determine which episode to replay
    if args.episode:
        episode_path = args.episode
        if not os.path.exists(episode_path):
            print(f"Error: Episode file not found: {episode_path}")
            return
    elif args.random:
        # Find all episodes
        episodes = glob.glob(os.path.join(args.data_dir, '*.npz'))
        if not episodes:
            print(f"Error: No episodes found in {args.data_dir}")
            return
        episode_path = random.choice(episodes)
        print(f"Randomly selected: {os.path.basename(episode_path)}\n")
    else:
        print("Error: Must specify --episode <path> or --random")
        print("\nExamples:")
        print("  python replay_episode.py --random")
        print("  python replay_episode.py --episode data/episodes/episode_1760374918_c96c401b.npz")
        return
    
    # Replay the episode
    replay_episode(episode_path, step_time=args.step_time, reset_first=not args.no_reset)


if __name__ == '__main__':
    main()
