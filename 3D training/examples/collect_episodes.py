"""Collect episodes for RL parking task.

This script connects to AirSim (Car), collects synchronized data per timestep and
saves episodes to disk. Each state vector is an embedding composed of 4 images
(front/rear/left/right) and two scalar values: distance to nearest obstacle and
distance to the parking goal.

Action format: (steering, throttle, brake)

Episode termination:
 - collision
 - distance to goal < goal_threshold

Notes:
 - Image embedding is a lightweight placeholder: resize + normalize + flatten.
   Replace with ViT embedding later by importing a pretrained ViT and calling it
   on each image crop or concatenated tensor.
 - Requires `numpy`, `opencv-python`, `airsim` (listed in requirements.txt).
"""

import os
import time
import uuid
import argparse
from typing import Tuple
import threading
import queue
from queue import Queue
import traceback

import numpy as np
import cv2
import airsim

# Try to import keyboard library for manual control
try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Warning: pynput not installed. Manual mode will not work.")
    print("Install with: pip install pynput")


class KeyboardController:
    """Simple keyboard controller for manual driving."""
    def __init__(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.active_keys = set()
        self.running = True
        
        if KEYBOARD_AVAILABLE:
            self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
            print("✓ Keyboard listener started")
            print("  Controls: Z/↑=Forward, S/↓=Brake, Q/←=Left, D/→=Right, SPACE=Emergency Brake")
    
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
        return self.steering, self.throttle, self.brake
    
    def stop(self):
        self.running = False
        if KEYBOARD_AVAILABLE and hasattr(self, 'listener'):
            self.listener.stop()


def image_embedding_placeholder(img: np.ndarray, size=(64, 64)) -> np.ndarray:
    """Simple image embedding: resize, convert to float32, normalize, flatten.
    Replace this with a ViT encoder for better performance.
    """
    im = cv2.resize(img, size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    return im.ravel()


def get_images(client: airsim.CarClient, camera_names) -> Tuple[np.ndarray, ...]:
    reqs = [airsim.ImageRequest(name, airsim.ImageType.Scene, False, False) for name in camera_names]
    responses = client.simGetImages(reqs, vehicle_name='Car1')
    imgs = []
    for r in responses:
        if r.height == 0:
            imgs.append(np.zeros((16, 16, 3), dtype=np.uint8))
            continue
        arr = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
        img = arr.reshape(r.height, r.width, 3)
        imgs.append(img)
    return tuple(imgs)


def _point_to_segment_distance(pt: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    """Compute the minimum distance from a 2D point to a line segment.
    
    Args:
        pt: point [x, y]
        seg_a: segment start [x, y]
        seg_b: segment end [x, y]
    
    Returns:
        Distance in meters
    """
    v = seg_b - seg_a
    w = pt - seg_a
    
    seg_len_sq = np.dot(v, v)
    if seg_len_sq < 1e-9:  # degenerate segment
        return float(np.linalg.norm(w))
    
    # project pt onto line; t is the parameter along the segment
    t = np.dot(w, v) / seg_len_sq
    
    if t < 0:
        # closest to seg_a
        return float(np.linalg.norm(w))
    elif t > 1:
        # closest to seg_b
        return float(np.linalg.norm(pt - seg_b))
    else:
        # closest to interior point on segment
        proj = seg_a + t * v
        return float(np.linalg.norm(pt - proj))


def _get_2d_bounding_box(position: airsim.Vector3r, orientation: airsim.Quaternionr, 
                          half_length: float, half_width: float) -> np.ndarray:
    """Compute the 4 corners of a rotated 2D bounding box (top-down XY plane).
    
    Returns:
        corners (4, 2): array of [x, y] for each corner
    """
    _, _, yaw = airsim.utils.to_eularian_angles(orientation)
    # AirSim yaw is in NED frame; adjust by rotating coordinate frame
    yaw = -(yaw + np.pi / 2)
    
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw],
                         [sin_yaw,  cos_yaw]])
    
    # Define corners in local frame: front-right, front-left, rear-left, rear-right
    local_corners = np.array([
        [ half_length,  half_width],
        [ half_length, -half_width],
        [-half_length, -half_width],
        [-half_length,  half_width]
    ])
    
    center = np.array([position.x_val, position.y_val])
    world_corners = (rotation @ local_corners.T).T + center
    
    return world_corners


def nearest_obstacle_distance_geometric(client: airsim.CarClient, 
                                        car_half_length=2.6, 
                                        car_half_width=1.05,
                                        obstacle_objects=None,
                                        _cached_objects=None) -> float:
    """Compute the geometric distance from the car to the nearest obstacle using 2D bounding boxes.
    
    This is far more reliable than depth images during collision, as it uses actual object poses
    and computes edge-to-edge distance.
    
    Args:
        client: AirSim car client
        car_half_length: half-length of the car bounding box (meters)
        car_half_width: half-width of the car bounding box (meters)
        obstacle_objects: list of object names to check (e.g., ['Car_2', 'Car_3']). 
                         If None, will auto-discover from scene.
    
    Returns:
        Distance to nearest obstacle edge in meters. Returns 0.0 if collision detected but no object found.
    """
    try:
        # Get car pose
        car_state = client.getCarState(vehicle_name='Car1')
        car_pos = car_state.kinematics_estimated.position
        car_orient = car_state.kinematics_estimated.orientation
        
        car_corners = _get_2d_bounding_box(car_pos, car_orient, car_half_length, car_half_width)
        
        # Auto-discover obstacles if not specified
        if obstacle_objects is None:
            # Use cached list if available (avoid repeated expensive API calls)
            if _cached_objects is None:
                try:
                    all_objects = client.simListSceneObjects()
                    # Filter out non-obstacle objects (keep things that might be obstacles)
                    exclude_keywords = ['sky', 'sun', 'light', 'camera', 'player', 'ground', 
                                       'landscape', 'floor', 'car1', 'atmospheric']
                    obstacle_objects = []
                    for obj in all_objects:
                        obj_lower = obj.lower()
                        if not any(kw in obj_lower for kw in exclude_keywords):
                            obstacle_objects.append(obj)
                    
                    # Cache for subsequent calls
                    if hasattr(nearest_obstacle_distance_geometric, '_object_cache'):
                        nearest_obstacle_distance_geometric._object_cache = obstacle_objects
                    
                    # print(f"[Auto-discovered {len(obstacle_objects)} potential obstacles: {obstacle_objects[:5]}...]")
                except Exception as e:
                    print(f"Failed to auto-discover objects: {e}")
                    # Fallback to common patterns
                    obstacle_objects = [f"Car_{i}" for i in range(2, 20)]
            else:
                obstacle_objects = _cached_objects
        
        min_dist = float('inf')
        found_any = False
        
        for obj_name in obstacle_objects:
            try:
                obs_pose = client.simGetObjectPose(obj_name)
                
                # Check if object exists and is not at origin (default invalid pose)
                pos_mag = (obs_pose.position.x_val**2 + obs_pose.position.y_val**2 + 
                          obs_pose.position.z_val**2)**0.5
                if pos_mag < 0.1:  # Too close to origin, likely invalid
                    continue
                
                found_any = True
                
                # Assume obstacles have similar dimensions (adjust if needed)
                obs_half_length = 2.65
                obs_half_width = 1.1
                
                obs_corners = _get_2d_bounding_box(obs_pose.position, obs_pose.orientation, 
                                                   obs_half_length, obs_half_width)
                
                # Compute minimum distance between all edges of car and obstacle
                for i in range(4):
                    for j in range(4):
                        # Car corners to obstacle edges
                        next_j = (j + 1) % 4
                        dist = _point_to_segment_distance(car_corners[i], obs_corners[j], obs_corners[next_j])
                        min_dist = min(min_dist, dist)
                        
                        # Obstacle corners to car edges
                        next_i = (i + 1) % 4
                        dist = _point_to_segment_distance(obs_corners[j], car_corners[i], car_corners[next_i])
                        min_dist = min(min_dist, dist)
                
            except Exception:
                # Object doesn't exist or pose query failed
                continue
        
        # If we're in collision but found no obstacles, return 0 (collision with unmapped object)
        coll = client.simGetCollisionInfo(vehicle_name='Car1')
        if coll.has_collided and (min_dist == float('inf') or not found_any):
            print(f"[WARNING] Collision detected but no obstacle objects found! Colliding with: {coll.object_name}")
            print(f"  Add '{coll.object_name}' to --obstacle_objects list")
            return 0.0
        
        return min_dist
        
    except Exception as e:
        print(f"Error computing geometric obstacle distance: {e}")
        return float('inf')


def distance_to_goal(car_pos: Tuple[float, float, float], goal_pos: Tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(car_pos) - np.array(goal_pos)))


def save_episode(episode_data: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"episode_{int(time.time())}_{uuid.uuid4().hex[:8]}.npz")
    # episode_data is expected to contain lists/arrays for states, actions, dones, rewards)
    np.savez_compressed(fname, **episode_data)
    print('Saved episode to', fname)


def collect_episode(client: airsim.CarClient, args, episode_index=0):
    camera_names = args.cameras

    goal_pos = np.array([args.goal_x, args.goal_y, args.goal_z], dtype=float)
    goal_threshold = args.goal_threshold

    episode_states = []  # list of np arrays (state vector)
    episode_actions = []  # list of (steer, throttle, brake)
    episode_rewards = []
    episode_dones = []
    
    # Print control mode for this episode
    if episode_index == 0:
        if args.auto:
            print("🤖 Control Mode: AUTO (random actions)")
        else:
            print("🎮 Control Mode: MANUAL")
            if KEYBOARD_AVAILABLE:
                print("   Drive using ZQSD or Arrow keys")
                print("   Z/↑=Forward, S/↓=Brake, Q/←=Left, D/→=Right, SPACE=Emergency Brake")
            else:
                print("   ERROR: pynput not installed!")
                print("   Install with: pip install pynput")
                return
    
    # Setup keyboard controller for manual mode
    keyboard_controller = None
    if not args.auto:
        if not KEYBOARD_AVAILABLE:
            print("Error: Cannot use manual mode without pynput. Install with: pip install pynput")
            return
        keyboard_controller = KeyboardController()
        time.sleep(0.5)  # Let keyboard listener initialize
    
    # ALWAYS enable API control for data collection
    # We control the car via API in both modes
    client.enableApiControl(True, vehicle_name='Car1')
    
    # Arm the vehicle (for some vehicle types this is required)
    # For cars, this usually isn't needed but doesn't hurt
    try:
        client.armDisarm(True, vehicle_name='Car1')
    except:
        pass  # Not all vehicle types support armDisarm

    # Store initial position for reset at end of episode (if enabled)
    initial_pose = None
    if args.reset_at_end:
        initial_state = client.getCarState(vehicle_name='Car1')
        initial_pose = airsim.Pose(
            initial_state.kinematics_estimated.position,
            initial_state.kinematics_estimated.orientation
        )
        if episode_index == 0:
            print(f"  Initial position stored: ({initial_pose.position.x_val:.1f}, {initial_pose.position.y_val:.1f}, {initial_pose.position.z_val:.1f})")

    # Check initial distance to goal
    initial_cs = client.getCarState(vehicle_name='Car1')
    initial_car_pos = (initial_cs.kinematics_estimated.position.x_val,
                      initial_cs.kinematics_estimated.position.y_val,
                      initial_cs.kinematics_estimated.position.z_val)
    initial_dist_to_goal = distance_to_goal(initial_car_pos, goal_pos)
    
    if episode_index == 0:
        print(f"  Starting distance to goal: {initial_dist_to_goal:.2f}m")
        if initial_dist_to_goal < goal_threshold:
            print(f"  ⚠️  WARNING: Already at goal! (dist={initial_dist_to_goal:.2f}m < threshold={goal_threshold}m)")
            print(f"  Episode will terminate immediately!")
        elif initial_dist_to_goal < 10.0:
            print(f"  ⚠️  WARNING: Very close to goal! Task might be too easy.")

    t0 = time.time()
    prev_dist = None

    # Simplified architecture: Fast control updates with periodic data collection
    # All in main thread to avoid AirSim IOLoop conflicts
    
    control_interval = 0.01  # 10ms = 100Hz control rate (ULTRA responsive!)
    data_collection_interval = args.step_time  # Collect data at original rate
    last_data_collection = time.time()
    step_counter = 0
    cached_speed = 0.0  # Use cached speed to avoid getCarState delay
    
    if episode_index == 0:
        print(f"  Control update rate: {1/control_interval:.0f} Hz (every {control_interval*1000:.0f}ms)")
        print(f"  Data collection rate: {1/data_collection_interval:.1f} Hz (every {data_collection_interval*1000:.0f}ms)")
    
    while True:
        # Step 1: Get keyboard input IMMEDIATELY (no API calls yet!)
        if args.auto:
            # Auto mode: keep using same action until it's time to collect data
            current_time = time.time()
            if step_counter == 0 or (current_time - last_data_collection >= data_collection_interval):
                steer = float(np.random.uniform(-1, 1))
                throttle = float(np.random.uniform(0, 1))
                brake = 0.0 if throttle > 0.05 else float(np.random.uniform(0, 1))
        else:
            # Manual mode: ALWAYS get fresh keyboard input
            steer, throttle, brake = keyboard_controller.get_controls()
        
        # Step 2: Apply controls IMMEDIATELY using cached speed for gear
        car_controls = airsim.CarControls()
        car_controls.steering = float(steer)
        car_controls.throttle = float(throttle)
        car_controls.brake = float(brake)
        
        # Automatic gear shifting using cached speed (good enough!)
        if throttle < 0:
            car_controls.is_manual_gear = True
            car_controls.manual_gear = -1
        elif throttle == 0 or brake > 0:
            car_controls.is_manual_gear = True
            car_controls.manual_gear = 0
        else:
            car_controls.is_manual_gear = True
            if cached_speed < 5:
                car_controls.manual_gear = 1
            elif cached_speed < 10:
                car_controls.manual_gear = 2
            elif cached_speed < 15:
                car_controls.manual_gear = 3
            else:
                car_controls.manual_gear = 4
        
        # SEND CONTROLS IMMEDIATELY!
        client.setCarControls(car_controls, vehicle_name='Car1')
        step_counter += 1
        
        # Step 3: Tiny sleep for control rate limiting
        time.sleep(control_interval)
        
        # Step 5: Only collect expensive data periodically
        current_time = time.time()
        if current_time - last_data_collection < data_collection_interval:
            continue  # Skip data collection, just keep applying controls
        
        last_data_collection = current_time
        
        # NOW do expensive operations (images, obstacles, etc.)
        # First, get current car state and update cached speed
        cs = client.getCarState(vehicle_name='Car1')
        vel = cs.kinematics_estimated.linear_velocity
        car_speed = float(np.linalg.norm([vel.x_val, vel.y_val, vel.z_val]))
        cached_speed = car_speed  # Update cache for gear shifting
        pos = cs.kinematics_estimated.position
        car_pos = (pos.x_val, pos.y_val, pos.z_val)
        
        # Get images
        imgs = get_images(client, camera_names)
        embeddings = [image_embedding_placeholder(im, size=(64, 64)) for im in imgs]
        img_emb = np.concatenate(embeddings)
        
        # Calculate obstacle distance
        nearest_obs = nearest_obstacle_distance_geometric(
            client, 
            car_half_length=args.car_half_length,
            car_half_width=args.car_half_width,
            obstacle_objects=args.obstacle_objects
        )
        
        # Distance to goal
        dist_goal = distance_to_goal(car_pos, goal_pos)
        
        # Build state vector
        state = np.concatenate([img_emb, np.array([nearest_obs, dist_goal, car_speed], dtype=np.float32)])
        
        # Get collision
        coll = client.simGetCollisionInfo(vehicle_name='Car1')
        done = coll.has_collided or (dist_goal <= goal_threshold)
        
        # Calculate reward
        step_penalty = float(args.step_penalty)
        success_bonus = float(args.success_bonus)
        collision_penalty = float(args.collision_penalty)

        if prev_dist is None:
            prev_dist = dist_goal

        progress = (prev_dist - dist_goal)
        reward = float(progress) - step_penalty

        if coll.has_collided:
            reward -= collision_penalty

        if (dist_goal <= goal_threshold) and (not coll.has_collided):
            reward += success_bonus

        prev_dist = dist_goal
        
        # Store data
        episode_states.append(state)
        episode_actions.append((steer, throttle, brake))
        episode_rewards.append(reward)
        episode_dones.append(done)
        
        # Debug output
        if len(episode_states) % 10 == 0 or len(episode_states) == 1:
            print(f"  Step {len(episode_states)}: pos=({car_pos[0]:.2f}, {car_pos[1]:.2f}), "
                  f"dist_goal={dist_goal:.2f}m, nearest_obs={nearest_obs:.2f}m, speed={car_speed:.2f}m/s")
        
        if coll.has_collided:
            print(f"\n⚠️  COLLISION at step {len(episode_states)}")
            print(f"  Object: '{coll.object_name}'\n")
        
        # Check termination
        if done:
            print(f'Episode finished: collision={coll.has_collided}, dist_goal={dist_goal:.3f}m')
            break

    # Cleanup keyboard controller
    if keyboard_controller is not None:
        keyboard_controller.stop()
    
    # Save
    epdata = {
        'states': np.stack(episode_states),
        'actions': np.array(episode_actions, dtype=np.float32),
        'rewards': np.array(episode_rewards, dtype=np.float32),
        'dones': np.array(episode_dones, dtype=np.uint8),
        'camera_names': np.array(camera_names)
    }
    save_episode(epdata, args.outdir)

    # Reset car to initial position if enabled
    if args.reset_at_end and initial_pose is not None:
        try:
            print(f"  Resetting car to initial position...")
            # Stop the car first
            stop_controls = airsim.CarControls()
            stop_controls.throttle = 0
            stop_controls.brake = 1.0
            client.setCarControls(stop_controls, vehicle_name='Car1')
            time.sleep(0.1)
            
            # Reset pose
            client.simSetVehiclePose(initial_pose, ignore_collision=True, vehicle_name='Car1')
            time.sleep(0.3)  # Wait for physics to settle
            print(f"  ✓ Car reset complete")
        except Exception as e:
            print(f"  ✗ Failed to reset car: {e}")


def collect_multiple(args):
    client = airsim.CarClient()
    client.confirmConnection()

    n = max(1, int(args.n_episodes))
    for i in range(n):
        print(f'Collecting episode {i+1}/{n}...')

        # Optionally reset vehicle pose before episode
        if args.reset_pose:
            try:
                pose = airsim.Pose(airsim.Vector3r(*args.reset_pose_xyz), airsim.to_quaternion(0,0,0))
                client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name='Car1')
                time.sleep(0.2)
            except Exception as e:
                print('Failed to reset pose:', e)

        collect_episode(client, args, episode_index=i)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', default='data/episodes')
    p.add_argument('--cameras', nargs='+', default=['0', 'rear', 'left', 'right'])
    p.add_argument('--goal_x', type=float, default=56.488)
    p.add_argument('--goal_y', type=float, default=3.012)
    p.add_argument('--goal_z', type=float, default=-0.639)
    p.add_argument('--goal_threshold', type=float, default=1.0)
    p.add_argument('--step_time', type=float, default=0.2)
    p.add_argument('--auto', action='store_true', help='Use automatic random actions. Without this flag, control car manually using AirSim built-in controls (WASD/Arrows in simulator)')
    p.add_argument('--n_episodes', type=int, default=1, help='Number of episodes to collect in this run')
    p.add_argument('--reset_pose', action='store_true', help='Reset vehicle pose before each episode')
    p.add_argument('--reset_pose_xyz', nargs=3, type=float, default=[0.0,0.0,0.0], help='XYZ for reset pose when --reset_pose is used')
    p.add_argument('--reset_at_end', action='store_true', help='Reset car to initial position at end of each episode')
    p.add_argument('--step_penalty', type=float, default=0.01, help='Per-step penalty to encourage shorter episodes')
    p.add_argument('--success_bonus', type=float, default=10.0, help='Reward bonus when the goal is reached')
    p.add_argument('--collision_penalty', type=float, default=50.0, help='Additional penalty applied on collision')
    p.add_argument('--car_half_length', type=float, default=2.6, help='Half-length of car bounding box (meters)')
    p.add_argument('--car_half_width', type=float, default=1.05, help='Half-width of car bounding box (meters)')
    p.add_argument('--obstacle_objects', nargs='+', default=None, help='List of obstacle object names (e.g., Car_2 Car_3). If not specified, will auto-detect.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    collect_multiple(args)
