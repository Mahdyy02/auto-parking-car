"""Record observation-action pairs for behavioural cloning.

Saves each recorded observation as a PNG and stores a single index pickle file listing
pairs: {'image': '<rel_path>', 'steering': float, 'throttle': float, 'brake': float, 'gear': int, ...}

Usage:
    python examples/record_bc_dataset.py --out data/bc_dataset/manual_drive

The script re-uses the same image preprocessing used by the RL env (84x84 grayscale) so
recorded observations are compatible with the CNN policies in the repo.
"""
import os
import argparse
import time
import pickle
from typing import List, Dict

import numpy as np
import airsim
from pynput import keyboard
from PIL import Image


class KeyboardController:
    """Captures keyboard input for manual car control (AZERTY/QWERTY friendly)."""

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
            if hasattr(key, 'char') and key.char is not None:
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
        except Exception:
            pass

        self.update_controls()

    def on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char is not None:
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
        except Exception:
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

        # Throttle/Brake: Z=forward, S=reverse, SPACE=full brake
        if 'space' in self.active_keys:
            self.throttle = 0.0
            self.brake = 1.0
        elif 'z' in self.active_keys and 's' not in self.active_keys:
            self.throttle = 1.0
            self.brake = 0.0
        elif 's' in self.active_keys and 'z' not in self.active_keys:
            self.throttle = -1.0
            self.brake = 0.0
        elif 'z' in self.active_keys and 's' in self.active_keys:
            self.throttle = 0.0
            self.brake = 0.0
        else:
            self.throttle = 0.0
            self.brake = 0.0

    def get_controls(self):
        return self.steering, self.throttle, self.brake

    def stop(self):
        self.listener.stop()


def transform_obs_from_airsim(response, out_shape=(84, 84)):
    """Replicate `AirSimCarEnv.transform_obs` preprocessing.

    Expects AirSim ImageResponse with image_data_float and height/width.
    Returns a uint8 grayscale image of shape (H, W).
    """
    img1d = np.array(response.image_data_float, dtype=np.float32)
    img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (response.height, response.width))

    image = Image.fromarray(img2d)
    im_final = np.array(image.resize(out_shape).convert('L'))
    return im_final.astype(np.uint8)


def calculate_gear(throttle: float, speed: float, brake: float) -> int:
    if throttle < 0:
        return -1
    elif brake > 0.5 or (abs(throttle) < 0.1 and brake > 0.1):
        return 0
    else:
        if speed < 5:
            return 1
        elif speed < 10:
            return 2
        elif speed < 15:
            return 3
        else:
            return 4


def record_bc_dataset(args):
    print("\n" + "=" * 60)
    print("BEHAVIOURAL CLONING DATA RECORDING")
    print("=" * 60)
    print("Drive manually; each recorded frame will be saved to disk.")
    print("Press Ctrl+C to stop recording and save the dataset.\n")

    # Connect to AirSim
    client = airsim.CarClient()
    client.confirmConnection()

    # Reset and prepare
    client.reset()
    time.sleep(0.5)
    client.enableApiControl(True, vehicle_name='Car1')
    try:
        client.armDisarm(True, vehicle_name='Car1')
    except Exception:
        pass

    # Release handbrake
    controls = airsim.CarControls()
    controls.handbrake = False
    client.setCarControls(controls, vehicle_name='Car1')
    time.sleep(0.05)

    # Create image request like the env
    image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)

    # Setup keyboard controller
    kb = KeyboardController()

    # Prepare output dirs
    images_dir = os.path.join(args.out, 'images')
    os.makedirs(images_dir, exist_ok=True)

    records: List[Dict] = []

    control_interval = args.control_interval
    recording_interval = args.recording_interval

    start_time = time.time()
    last_record_time = start_time
    cached_speed = 0.0
    frame_count = 0

    print("Recording... ctrl-C to stop.\n")
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            # get keyboard state
            steer, throttle, brake = kb.get_controls()

            # calculate gear using cached speed
            gear = calculate_gear(throttle, cached_speed, brake)

            # apply controls immediately
            car_controls = airsim.CarControls()
            car_controls.steering = float(steer)
            car_controls.throttle = float(throttle)
            car_controls.brake = float(brake)
            car_controls.handbrake = False
            car_controls.is_manual_gear = True
            car_controls.manual_gear = gear
            client.setCarControls(car_controls, vehicle_name='Car1')

            # sleep for control tick
            time.sleep(control_interval)

            # periodically record observation and action
            if current_time - last_record_time >= recording_interval:
                # fetch image + state
                responses = client.simGetImages([image_request])
                img = transform_obs_from_airsim(responses[0], out_shape=tuple(args.image_shape[:2]))

                # get car state
                car_state = client.getCarState(vehicle_name='Car1')
                cached_speed = car_state.speed
                pos = car_state.kinematics_estimated.position
                car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])

                # store image as PNG
                fname = f"frame_{int(elapsed*1000):08d}_{frame_count:06d}.png"
                fpath = os.path.join(images_dir, fname)
                Image.fromarray(img).save(fpath)

                # append record
                rec = {
                    'image': os.path.relpath(fpath, args.out),
                    'steering': float(steer),
                    'throttle': float(throttle),
                    'brake': float(brake),
                    'gear': int(gear),
                    'speed': float(cached_speed),
                    'position': car_pos.tolist(),
                    'timestamp': float(elapsed),
                }
                records.append(rec)

                frame_count += 1
                last_record_time = current_time

                if frame_count % 50 == 0:
                    print(f"Recorded {frame_count} frames, t={elapsed:.1f}s, speed={cached_speed:.2f} m/s")

                # auto stop
                if args.auto_stop:
                    # compute distance to hardcoded default goal similar to examples
                    goal_pos = np.array([56.49, 3.01, -0.64])
                    dist = np.linalg.norm(car_pos - goal_pos)
                    collision = client.simGetCollisionInfo(vehicle_name='Car1').has_collided
                    if collision:
                        print("Collision detected — stopping recording")
                        break
                    if dist <= 2.0:
                        print("Goal reached — stopping recording")
                        break

            # check max duration
            if args.max_duration and elapsed >= args.max_duration:
                print(f"Max duration {args.max_duration}s reached — stopping")
                break

    except KeyboardInterrupt:
        print("\nRecording stopped by user")

    finally:
        # stop the car
        stop_c = airsim.CarControls()
        stop_c.throttle = 0
        stop_c.brake = 1.0
        client.setCarControls(stop_c, vehicle_name='Car1')
        kb.stop()

    # Save index file
    dataset = {
        'records': records,
        'total_duration': elapsed,
        'frame_count': frame_count,
        'image_shape': args.image_shape,
        'recording_interval': recording_interval,
        'control_interval': control_interval,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, 'index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n✓ Saved dataset to {args.out}")
    print(f"  frames: {frame_count}, duration: {elapsed:.1f}s")
    print("You can use this folder directly for behavioural cloning training.")


def main():
    parser = argparse.ArgumentParser(description='Record BC dataset from manual driving')
    parser.add_argument('--out', type=str, default='data/bc_dataset/manual_drive',
                        help='Output folder for dataset (images/ + index.pkl)')
    parser.add_argument('--image-shape', type=int, nargs=3, default=(84, 84, 1),
                        help='Image shape to save (H W C). Only HxW used; default 84 84 1')
    parser.add_argument('--control-interval', type=float, default=0.01,
                        help='Control loop interval in seconds (default 0.01 = 100Hz)')
    parser.add_argument('--recording-interval', type=float, default=0.05,
                        help='Recording interval in seconds (default 0.05 = 20Hz)')
    parser.add_argument('--max-duration', type=float, default=None,
                        help='Maximum recording duration in seconds')
    parser.add_argument('--auto-stop', action='store_true',
                        help='Automatically stop when goal reached or collision occurs')
    args = parser.parse_args()

    record_bc_dataset(args)


if __name__ == '__main__':
    main()
