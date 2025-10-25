"""Record a manual driving trajectory and save it as a sequence of actions.

This script records your manual driving as a simple list of actions that can be
replayed exactly without any neural network training.

Usage:
    python record_trajectory.py --output data/trajectories/my_path.pkl
    
    # With auto-stop when goal reached
    python record_trajectory.py --output data/trajectories/my_path.pkl --auto_stop
    
    # With time limit
    python record_trajectory.py --output data/trajectories/my_path.pkl --max_duration 120
"""
import os
import argparse
import time
import pickle
from typing import List, Dict

import numpy as np
import airsim
from pynput import keyboard


class KeyboardController:
    """Captures keyboard input for manual car control (AZERTY layout)."""
    
    def __init__(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.active_keys = set()
        
        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
        print("✓ Keyboard listener started")
        print("  Controls: Z/↑=Forward, S/↓=Reverse, Q/←=Left, D/→=Right, SPACE=Brake")
    
    def on_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char in ['z', 'Z']:
                    self.active_keys.add('z')
                elif key.char in ['s', 'S']:
                    self.active_keys.add('s')
                elif key.char in ['q', 'Q']:
                    self.active_keys.add('q')
                elif key.char in ['d', 'D']:
                    self.active_keys.add('d')
            else:
                if key == keyboard.Key.up:
                    self.active_keys.add('z')
                elif key == keyboard.Key.down:
                    self.active_keys.add('s')
                elif key == keyboard.Key.left:
                    self.active_keys.add('q')
                elif key == keyboard.Key.right:
                    self.active_keys.add('d')
                elif key == keyboard.Key.space:
                    self.active_keys.add('space')
        except:
            pass
        
        self.update_controls()
    
    def on_release(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char in ['z', 'Z']:
                    self.active_keys.discard('z')
                elif key.char in ['s', 'S']:
                    self.active_keys.discard('s')
                elif key.char in ['q', 'Q']:
                    self.active_keys.discard('q')
                elif key.char in ['d', 'D']:
                    self.active_keys.discard('d')
            else:
                if key == keyboard.Key.up:
                    self.active_keys.discard('z')
                elif key == keyboard.Key.down:
                    self.active_keys.discard('s')
                elif key == keyboard.Key.left:
                    self.active_keys.discard('q')
                elif key == keyboard.Key.right:
                    self.active_keys.discard('d')
                elif key == keyboard.Key.space:
                    self.active_keys.discard('space')
        except:
            pass
        
        self.update_controls()
    
    def update_controls(self):
        """Update control values based on currently pressed keys."""
        # Steering: Q=left, D=right
        if 'q' in self.active_keys and 'd' not in self.active_keys:
            self.steering = -0.5
        elif 'd' in self.active_keys and 'q' not in self.active_keys:
            self.steering = 0.5
        else:
            self.steering = 0.0
        
        # Throttle/Brake: Z=forward, S=reverse (backward), SPACE=full brake
        if 'space' in self.active_keys:
            # SPACE = emergency brake (highest priority)
            self.throttle = 0.0
            self.brake = 1.0
        elif 'z' in self.active_keys and 's' not in self.active_keys:
            # Z only = forward
            self.throttle = 1.0
            self.brake = 0.0
        elif 's' in self.active_keys and 'z' not in self.active_keys:
            # S only = reverse (backward movement)
            self.throttle = -1.0
            self.brake = 0.0
        elif 'z' in self.active_keys and 's' in self.active_keys:
            # Both pressed = cancel out (no movement)
            self.throttle = 0.0
            self.brake = 0.0
        else:
            # Nothing pressed = coast
            self.throttle = 0.0
            self.brake = 0.0
    
    def get_controls(self):
        """Get current control values based on pressed keys."""
        return self.steering, self.throttle, self.brake
    
    def stop(self):
        """Stop the keyboard listener."""
        self.listener.stop()


def calculate_gear(throttle: float, speed: float, brake: float) -> int:
    """Calculate appropriate gear based on throttle, speed, and brake."""
    if throttle < 0:
        return -1  # Reverse
    elif brake > 0.5 or (abs(throttle) < 0.1 and brake > 0.1):
        return 0  # Neutral
    else:
        # Forward gears
        if speed < 5:
            return 1
        elif speed < 10:
            return 2
        elif speed < 15:
            return 3
        else:
            return 4


def record_trajectory(args):
    """Record manual driving trajectory."""
    print("\n" + "="*80)
    print("TRAJECTORY RECORDING")
    print("="*80)
    print("\nControls (AZERTY keyboard):")
    print("  Z - Forward")
    print("  S - Reverse")
    print("  Q - Turn left")
    print("  D - Turn right")
    print("  SPACE - Brake")
    print("\nPress Ctrl+C to stop recording\n")
    
    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.CarClient()
    client.confirmConnection()
    print("✓ Connected\n")
    
    # Reset environment
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
    
    # Get goal position (from settings or default)
    goal_pos = np.array([56.49, 3.01, -0.64])  # Default parking spot
    
    # Initialize keyboard controller
    keyboard_controller = KeyboardController()
    
    # Storage for trajectory
    trajectory: List[Dict] = []
    
    # Recording parameters
    control_interval = 0.01  # 100 Hz control rate
    recording_interval = 0.05  # 20 Hz recording rate (record every 5th control step)
    
    start_time = time.time()
    last_record_time = start_time
    cached_speed = 0.0
    
    print("="*80)
    print("RECORDING STARTED")
    print("="*80)
    print("Drive to the goal. Press Ctrl+C when finished.\n")
    
    try:
        step_count = 0
        while True:
            step_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check time limit
            if args.max_duration and elapsed_time > args.max_duration:
                print(f"\n✓ Max duration ({args.max_duration}s) reached")
                break
            
            # Get keyboard input
            steer, throttle, brake = keyboard_controller.get_controls()
            
            # Calculate gear using cached speed
            gear = calculate_gear(throttle, cached_speed, brake)
            
            # Apply controls immediately
            car_controls = airsim.CarControls()
            car_controls.steering = float(steer)
            car_controls.throttle = float(throttle)
            car_controls.brake = float(brake)
            car_controls.handbrake = False
            car_controls.is_manual_gear = True
            car_controls.manual_gear = gear
            
            client.setCarControls(car_controls, vehicle_name='Car1')
            
            # Sleep for control interval
            time.sleep(control_interval)
            
            # Record data periodically (not every step to reduce size)
            if current_time - last_record_time >= recording_interval:
                # Get car state
                car_state = client.getCarState(vehicle_name='Car1')
                cached_speed = car_state.speed
                
                pos = car_state.kinematics_estimated.position
                car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
                
                # Calculate distance to goal
                dist_to_goal = np.linalg.norm(car_pos - goal_pos)
                
                # Check collision
                collision_info = client.simGetCollisionInfo(vehicle_name='Car1')
                has_collided = collision_info.has_collided
                
                # Store action and metadata
                action_data = {
                    'steering': float(steer),
                    'throttle': float(throttle),
                    'brake': float(brake),
                    'gear': int(gear),
                    'speed': float(cached_speed),
                    'position': car_pos.tolist(),
                    'distance_to_goal': float(dist_to_goal),
                    'timestamp': float(elapsed_time)
                }
                
                trajectory.append(action_data)
                last_record_time = current_time
                
                # Print status
                if len(trajectory) % 20 == 0:
                    print(f"Recording: {elapsed_time:.1f}s, "
                          f"{len(trajectory)} actions, "
                          f"speed={cached_speed:.2f} m/s, "
                          f"dist={dist_to_goal:.2f}m")
                
                # Check auto-stop conditions
                if args.auto_stop:
                    if has_collided:
                        print(f"\n✗ Collision detected - stopping recording")
                        break
                    if dist_to_goal <= 2.0:
                        print(f"\n✓ Goal reached - stopping recording")
                        break
    
    except KeyboardInterrupt:
        print("\n✓ Recording stopped by user")
    
    finally:
        # Stop the car
        stop_controls = airsim.CarControls()
        stop_controls.throttle = 0
        stop_controls.brake = 1.0
        client.setCarControls(stop_controls, vehicle_name='Car1')
        
        # Stop keyboard listener
        keyboard_controller.stop()
    
    # Save trajectory
    print(f"\n{'='*80}")
    print(f"RECORDING COMPLETE")
    print(f"{'='*80}")
    print(f"Total duration:  {elapsed_time:.2f} seconds")
    print(f"Total actions:   {len(trajectory)}")
    print(f"Recording rate:  {len(trajectory)/elapsed_time:.1f} Hz")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save trajectory data
    trajectory_data = {
        'actions': trajectory,
        'goal_position': goal_pos.tolist(),
        'total_duration': elapsed_time,
        'recording_rate': len(trajectory) / elapsed_time if elapsed_time > 0 else 0,
        'metadata': {
            'recording_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'control_interval': control_interval,
            'recording_interval': recording_interval,
        }
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(trajectory_data, f)
    
    print(f"\n✓ Trajectory saved to: {args.output}")
    print(f"\nYou can replay this trajectory with:")
    print(f"  python replay_trajectory.py --trajectory {args.output}")


def main():
    parser = argparse.ArgumentParser(description='Record manual driving trajectory')
    parser.add_argument('--output', type=str, default='data/trajectories/manual_trajectory.pkl',
                        help='Output file for trajectory (default: data/trajectories/manual_trajectory.pkl)')
    parser.add_argument('--auto_stop', action='store_true',
                        help='Automatically stop when goal reached or collision occurs')
    parser.add_argument('--max_duration', type=float, default=None,
                        help='Maximum recording duration in seconds (default: unlimited)')
    
    args = parser.parse_args()
    
    record_trajectory(args)


if __name__ == '__main__':
    main()
