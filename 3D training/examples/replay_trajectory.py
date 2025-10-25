<<<<<<< HEAD
"""Replay a recorded trajectory in AirSim.

This script replays the exact sequence of actions you recorded, without any neural network.

Usage:
    python replay_trajectory.py --trajectory data/trajectories/manual_trajectory.pkl
    
    # Replay multiple times
    python replay_trajectory.py --trajectory data/trajectories/manual_trajectory.pkl --n_replays 5
    
    # Slower/faster playback
    python replay_trajectory.py --trajectory data/trajectories/manual_trajectory.pkl --speed 0.5  # Half speed
    python replay_trajectory.py --trajectory data/trajectories/manual_trajectory.pkl --speed 2.0  # Double speed
"""
import argparse
import time
import pickle

import numpy as np
import airsim


def replay_trajectory(client, trajectory_data, args):
    """Replay a single trajectory."""
    
    actions = trajectory_data['actions']
    goal_pos = np.array(trajectory_data['goal_position'])
    
    print(f"\n{'='*80}")
    print(f"REPLAYING TRAJECTORY")
    print(f"{'='*80}")
    print(f"Total actions:    {len(actions)}")
    print(f"Duration:         {trajectory_data['total_duration']:.2f}s")
    print(f"Playback speed:   {args.speed}x")
    print(f"Goal position:    ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})")
    print()
    
    # Reset environment
    if args.reset:
        client.reset()
        time.sleep(1.0)
    
    # Enable API control
    client.enableApiControl(True, vehicle_name='Car1')
    try:
        client.armDisarm(True, vehicle_name='Car1')
    except:
        pass
    
    # Release handbrake
    initial_controls = airsim.CarControls()
    initial_controls.handbrake = False
    client.setCarControls(initial_controls, vehicle_name='Car1')
    time.sleep(0.1)
    
    # Get initial state
    car_state = client.getCarState(vehicle_name='Car1')
    pos = car_state.kinematics_estimated.position
    car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
    initial_dist = np.linalg.norm(car_pos - goal_pos)
    
    print(f"Starting position: ({car_pos[0]:.2f}, {car_pos[1]:.2f}, {car_pos[2]:.2f})")
    print(f"Initial distance:  {initial_dist:.2f} m")
    print(f"\nStarting replay...\n")
    
    # Replay actions
    start_time = time.time()
    prev_timestamp = 0.0
    
    outcome = "COMPLETED"
    
    for i, action in enumerate(actions):
        # Calculate delay based on recorded timestamps
        if i > 0:
            time_diff = action['timestamp'] - prev_timestamp
            # Apply speed multiplier
            adjusted_delay = time_diff / args.speed
            time.sleep(max(0, adjusted_delay))
        
        prev_timestamp = action['timestamp']
        
        # Apply the recorded action
        controls = airsim.CarControls()
        controls.steering = action['steering']
        controls.throttle = action['throttle']
        controls.brake = action['brake']
        controls.handbrake = False
        controls.is_manual_gear = True
        controls.manual_gear = action['gear']
        
        client.setCarControls(controls, vehicle_name='Car1')
        
        # Get current state
        car_state = client.getCarState(vehicle_name='Car1')
        pos = car_state.kinematics_estimated.position
        car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        current_speed = car_state.speed
        dist_to_goal = np.linalg.norm(car_pos - goal_pos)
        
        # Check collision
        collision_info = client.simGetCollisionInfo(vehicle_name='Car1')
        has_collided = collision_info.has_collided
        
        # Print progress
        if (i + 1) % 20 == 0 or (i + 1) == len(actions):
            elapsed = time.time() - start_time
            print(f"  Action {i+1:4d}/{len(actions)}: "
                  f"speed={current_speed:5.2f} m/s, "
                  f"dist={dist_to_goal:6.2f}m, "
                  f"steer={action['steering']:+.2f}, "
                  f"throttle={action['throttle']:+.2f}, "
                  f"brake={action['brake']:.2f}")
        
        # Check termination
        if has_collided:
            outcome = "COLLISION ✗"
            print(f"\n  ✗ Collision detected at action {i+1}")
            break
        
        if dist_to_goal <= 2.0:
            outcome = "SUCCESS ✓"
            print(f"\n  ✓ Goal reached at action {i+1}")
            break
    
    # Stop the car
    stop_controls = airsim.CarControls()
    stop_controls.throttle = 0
    stop_controls.brake = 1.0
    client.setCarControls(stop_controls, vehicle_name='Car1')
    
    # Final stats
    total_time = time.time() - start_time
    car_state = client.getCarState(vehicle_name='Car1')
    pos = car_state.kinematics_estimated.position
    final_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
    final_dist = np.linalg.norm(final_pos - goal_pos)
    
    print(f"\n{'='*80}")
    print(f"REPLAY COMPLETE")
    print(f"{'='*80}")
    print(f"Outcome:          {outcome}")
    print(f"Actions executed: {i+1}/{len(actions)}")
    print(f"Replay time:      {total_time:.2f}s")
    print(f"Final position:   ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
    print(f"Final distance:   {final_dist:.2f}m")
    print(f"Distance traveled: {initial_dist - final_dist:.2f}m")
    print()
    
    return {
        'outcome': outcome,
        'actions_executed': i + 1,
        'total_actions': len(actions),
        'final_distance': final_dist,
        'distance_traveled': initial_dist - final_dist,
        'success': outcome == "SUCCESS ✓"
    }


def main():
    parser = argparse.ArgumentParser(description='Replay recorded trajectory')
    parser.add_argument('--trajectory', type=str, required=True,
                        help='Path to recorded trajectory file (.pkl)')
    parser.add_argument('--n_replays', type=int, default=1,
                        help='Number of times to replay trajectory (default: 1)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset environment before each replay')
    
    args = parser.parse_args()
    
    # Load trajectory
    print(f"\nLoading trajectory from {args.trajectory}...")
    with open(args.trajectory, 'rb') as f:
        trajectory_data = pickle.load(f)
    
    print(f"✓ Loaded trajectory with {len(trajectory_data['actions'])} actions")
    print(f"  Recorded on: {trajectory_data['metadata']['recording_date']}")
    print(f"  Duration: {trajectory_data['total_duration']:.2f}s")
    
    # Connect to AirSim
    print(f"\nConnecting to AirSim...")
    client = airsim.CarClient()
    client.confirmConnection()
    print("✓ Connected")
    
    # Replay trajectory multiple times
    results = []
    for replay_num in range(args.n_replays):
        if args.n_replays > 1:
            print(f"\n{'='*80}")
            print(f"REPLAY {replay_num + 1}/{args.n_replays}")
            print(f"{'='*80}")
        
        result = replay_trajectory(client, trajectory_data, args)
        results.append(result)
        
        if replay_num < args.n_replays - 1:
            print("\nWaiting 2 seconds before next replay...")
            time.sleep(2.0)
    
    # Summary for multiple replays
    if args.n_replays > 1:
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY ({args.n_replays} replays)")
        print(f"{'='*80}")
        successes = sum(1 for r in results if r['success'])
        print(f"Success rate: {successes}/{args.n_replays} ({100*successes/args.n_replays:.1f}%)")
        avg_dist = np.mean([r['final_distance'] for r in results])
        print(f"Average final distance: {avg_dist:.2f}m")
    
    # Disable API control
    client.enableApiControl(False, vehicle_name='Car1')


if __name__ == '__main__':
    main()
=======
"""Replay a recorded trajectory in AirSim.

This script replays the exact sequence of actions you recorded, without any neural network.

Usage:
    python replay_trajectory.py --trajectory data/trajectories/manual_trajectory.pkl
    
    # Replay multiple times
    python replay_trajectory.py --trajectory data/trajectories/manual_trajectory.pkl --n_replays 5
    
    # Slower/faster playback
    python replay_trajectory.py --trajectory data/trajectories/manual_trajectory.pkl --speed 0.5  # Half speed
    python replay_trajectory.py --trajectory data/trajectories/manual_trajectory.pkl --speed 2.0  # Double speed
"""
import argparse
import time
import pickle

import numpy as np
import airsim


def replay_trajectory(client, trajectory_data, args):
    """Replay a single trajectory."""
    
    actions = trajectory_data['actions']
    goal_pos = np.array(trajectory_data['goal_position'])
    
    print(f"\n{'='*80}")
    print(f"REPLAYING TRAJECTORY")
    print(f"{'='*80}")
    print(f"Total actions:    {len(actions)}")
    print(f"Duration:         {trajectory_data['total_duration']:.2f}s")
    print(f"Playback speed:   {args.speed}x")
    print(f"Goal position:    ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})")
    print()
    
    # Reset environment
    if args.reset:
        client.reset()
        time.sleep(1.0)
    
    # Enable API control
    client.enableApiControl(True, vehicle_name='Car1')
    try:
        client.armDisarm(True, vehicle_name='Car1')
    except:
        pass
    
    # Release handbrake
    initial_controls = airsim.CarControls()
    initial_controls.handbrake = False
    client.setCarControls(initial_controls, vehicle_name='Car1')
    time.sleep(0.1)
    
    # Get initial state
    car_state = client.getCarState(vehicle_name='Car1')
    pos = car_state.kinematics_estimated.position
    car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
    initial_dist = np.linalg.norm(car_pos - goal_pos)
    
    print(f"Starting position: ({car_pos[0]:.2f}, {car_pos[1]:.2f}, {car_pos[2]:.2f})")
    print(f"Initial distance:  {initial_dist:.2f} m")
    print(f"\nStarting replay...\n")
    
    # Replay actions
    start_time = time.time()
    prev_timestamp = 0.0
    
    outcome = "COMPLETED"
    
    for i, action in enumerate(actions):
        # Calculate delay based on recorded timestamps
        if i > 0:
            time_diff = action['timestamp'] - prev_timestamp
            # Apply speed multiplier
            adjusted_delay = time_diff / args.speed
            time.sleep(max(0, adjusted_delay))
        
        prev_timestamp = action['timestamp']
        
        # Apply the recorded action
        controls = airsim.CarControls()
        controls.steering = action['steering']
        controls.throttle = action['throttle']
        controls.brake = action['brake']
        controls.handbrake = False
        controls.is_manual_gear = True
        controls.manual_gear = action['gear']
        
        client.setCarControls(controls, vehicle_name='Car1')
        
        # Get current state
        car_state = client.getCarState(vehicle_name='Car1')
        pos = car_state.kinematics_estimated.position
        car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        current_speed = car_state.speed
        dist_to_goal = np.linalg.norm(car_pos - goal_pos)
        
        # Check collision
        collision_info = client.simGetCollisionInfo(vehicle_name='Car1')
        has_collided = collision_info.has_collided
        
        # Print progress
        if (i + 1) % 20 == 0 or (i + 1) == len(actions):
            elapsed = time.time() - start_time
            print(f"  Action {i+1:4d}/{len(actions)}: "
                  f"speed={current_speed:5.2f} m/s, "
                  f"dist={dist_to_goal:6.2f}m, "
                  f"steer={action['steering']:+.2f}, "
                  f"throttle={action['throttle']:+.2f}, "
                  f"brake={action['brake']:.2f}")
        
        # Check termination
        if has_collided:
            outcome = "COLLISION ✗"
            print(f"\n  ✗ Collision detected at action {i+1}")
            break
        
        if dist_to_goal <= 2.0:
            outcome = "SUCCESS ✓"
            print(f"\n  ✓ Goal reached at action {i+1}")
            break
    
    # Stop the car
    stop_controls = airsim.CarControls()
    stop_controls.throttle = 0
    stop_controls.brake = 1.0
    client.setCarControls(stop_controls, vehicle_name='Car1')
    
    # Final stats
    total_time = time.time() - start_time
    car_state = client.getCarState(vehicle_name='Car1')
    pos = car_state.kinematics_estimated.position
    final_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
    final_dist = np.linalg.norm(final_pos - goal_pos)
    
    print(f"\n{'='*80}")
    print(f"REPLAY COMPLETE")
    print(f"{'='*80}")
    print(f"Outcome:          {outcome}")
    print(f"Actions executed: {i+1}/{len(actions)}")
    print(f"Replay time:      {total_time:.2f}s")
    print(f"Final position:   ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
    print(f"Final distance:   {final_dist:.2f}m")
    print(f"Distance traveled: {initial_dist - final_dist:.2f}m")
    print()
    
    return {
        'outcome': outcome,
        'actions_executed': i + 1,
        'total_actions': len(actions),
        'final_distance': final_dist,
        'distance_traveled': initial_dist - final_dist,
        'success': outcome == "SUCCESS ✓"
    }


def main():
    parser = argparse.ArgumentParser(description='Replay recorded trajectory')
    parser.add_argument('--trajectory', type=str, required=True,
                        help='Path to recorded trajectory file (.pkl)')
    parser.add_argument('--n_replays', type=int, default=1,
                        help='Number of times to replay trajectory (default: 1)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset environment before each replay')
    
    args = parser.parse_args()
    
    # Load trajectory
    print(f"\nLoading trajectory from {args.trajectory}...")
    with open(args.trajectory, 'rb') as f:
        trajectory_data = pickle.load(f)
    
    print(f"✓ Loaded trajectory with {len(trajectory_data['actions'])} actions")
    print(f"  Recorded on: {trajectory_data['metadata']['recording_date']}")
    print(f"  Duration: {trajectory_data['total_duration']:.2f}s")
    
    # Connect to AirSim
    print(f"\nConnecting to AirSim...")
    client = airsim.CarClient()
    client.confirmConnection()
    print("✓ Connected")
    
    # Replay trajectory multiple times
    results = []
    for replay_num in range(args.n_replays):
        if args.n_replays > 1:
            print(f"\n{'='*80}")
            print(f"REPLAY {replay_num + 1}/{args.n_replays}")
            print(f"{'='*80}")
        
        result = replay_trajectory(client, trajectory_data, args)
        results.append(result)
        
        if replay_num < args.n_replays - 1:
            print("\nWaiting 2 seconds before next replay...")
            time.sleep(2.0)
    
    # Summary for multiple replays
    if args.n_replays > 1:
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY ({args.n_replays} replays)")
        print(f"{'='*80}")
        successes = sum(1 for r in results if r['success'])
        print(f"Success rate: {successes}/{args.n_replays} ({100*successes/args.n_replays:.1f}%)")
        avg_dist = np.mean([r['final_distance'] for r in results])
        print(f"Average final distance: {avg_dist:.2f}m")
    
    # Disable API control
    client.enableApiControl(False, vehicle_name='Car1')


if __name__ == '__main__':
    main()
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
