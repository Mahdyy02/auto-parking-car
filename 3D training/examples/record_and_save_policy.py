<<<<<<< HEAD
"""Record a reproducible deterministic policy from manual driving.

This script records (state -> action) pairs using the same state construction used
by `examples/eval_policy.py` (image embedding placeholder + scalars), trains a
DeterministicPolicy (MLP backbone + action_head), and saves a checkpoint in the
format `{'policy_state_dict': ..., 'state_dim': ...}` which `eval_policy.py` can load.

Usage:
    python examples/record_and_save_policy.py --out data/manual_policy/policy_manual.pt

Notes:
- Reproducible: sets RNG seeds for torch/numpy/random and disables CuDNN non-determinism.
- State construction: uses the same `image_embedding_placeholder` embedding as `eval_policy.py`.
"""
import os
import time
import argparse
import random
from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import airsim

# ----------------------------- Reproducibility -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic CUDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------- Policy class (match eval_policy) -------------
class DeterministicPolicy(nn.Module):
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
        steering = torch.tanh(actions[:, 0:1])
        throttle = torch.tanh(actions[:, 1:2])
        brake = torch.sigmoid(actions[:, 2:3])
        return torch.cat([steering, throttle, brake], dim=1)

# ----------------------------- State construction helpers -------------------

def image_embedding_placeholder(img: np.ndarray, size=(64, 64)) -> np.ndarray:
    """Same embedding used by eval_policy: resize + RGB flattening.

    Expects an HxWx3 RGB image (uint8). Returns flattened float32 embedding.
    """
    im = cv2.resize(img, size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    return im.ravel()


def get_images(client: airsim.CarClient, camera_names: List[str], vehicle_name='Car1') -> Tuple[np.ndarray, ...]:
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


def get_state(client: airsim.CarClient, camera_names: List[str], goal_pos: np.ndarray, vehicle_name='Car1'):
    imgs = get_images(client, camera_names, vehicle_name)
    embeddings = [image_embedding_placeholder(im, size=(64, 64)) for im in imgs]
    img_emb = np.concatenate(embeddings)

    cs = client.getCarState(vehicle_name=vehicle_name)
    pos = cs.kinematics_estimated.position
    car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])

    vel = cs.kinematics_estimated.linear_velocity
    car_speed = float(np.linalg.norm([vel.x_val, vel.y_val, vel.z_val]))

    dist_to_goal = float(np.linalg.norm(car_pos - goal_pos))

    # placeholder for obstacle distance (keep same default as eval_policy)
    dist_to_obstacle = 10.0

    state = np.concatenate([img_emb, [dist_to_obstacle], [dist_to_goal], [car_speed]])
    return state, car_pos, dist_to_goal, car_speed

# ----------------------------- Dataset & training ---------------------------
class StateActionDataset(Dataset):
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        assert states.shape[0] == actions.shape[0]
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# ----------------------------- Recording + Train workflow -------------------

def record_states(args) -> Tuple[np.ndarray, np.ndarray]:
    print("Connecting to AirSim...")
    client = airsim.CarClient()
    client.confirmConnection()
    print("Connected")

    client.reset()
    time.sleep(0.5)
    client.enableApiControl(True, vehicle_name='Car1')
    try:
        client.armDisarm(True, vehicle_name='Car1')
    except Exception:
        pass

    # release handbrake
    c = airsim.CarControls()
    c.handbrake = False
    client.setCarControls(c, vehicle_name='Car1')
    time.sleep(0.05)

    camera_names = args.cameras
    goal_pos = np.array([args.goal_x, args.goal_y, args.goal_z], dtype=float)

    print("Start driving - press Ctrl+C to stop recording")
    states = []
    actions = []
    start_time = time.time()
    last_record = start_time
    cached_speed = 0.0

    # keyboard control similar to examples/record_trajectory.py
    from pynput import keyboard
    active_keys = set()

    def on_press(key):
        try:
            if hasattr(key, 'char') and key.char:
                active_keys.add(key.char.lower())
            else:
                if key == keyboard.Key.up:
                    active_keys.add('up')
                elif key == keyboard.Key.down:
                    active_keys.add('down')
                elif key == keyboard.Key.left:
                    active_keys.add('left')
                elif key == keyboard.Key.right:
                    active_keys.add('right')
                elif key == keyboard.Key.space:
                    active_keys.add('space')
        except Exception:
            pass

    def on_release(key):
        try:
            if hasattr(key, 'char') and key.char:
                active_keys.discard(key.char.lower())
            else:
                if key == keyboard.Key.up:
                    active_keys.discard('up')
                elif key == keyboard.Key.down:
                    active_keys.discard('down')
                elif key == keyboard.Key.left:
                    active_keys.discard('left')
                elif key == keyboard.Key.right:
                    active_keys.discard('right')
                elif key == keyboard.Key.space:
                    active_keys.discard('space')
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            # compute simple controls from keys
            # steering
            steer = 0.0
            if 'q' in active_keys or 'left' in active_keys:
                steer = -0.5
            if 'd' in active_keys or 'right' in active_keys:
                steer = 0.5
            # throttle/brake
            if ' ' in active_keys:
                throttle = 0.0
                brake = 1.0
            elif 'z' in active_keys or 'up' in active_keys:
                throttle = 1.0
                brake = 0.0
            elif 's' in active_keys or 'down' in active_keys:
                throttle = -1.0
                brake = 0.0
            else:
                throttle = 0.0
                brake = 0.0

            # compute gear (simple)
            if throttle < 0:
                gear = -1
            elif brake > 0.5 or (abs(throttle) < 0.1 and brake > 0.1):
                gear = 0
            else:
                if cached_speed < 5:
                    gear = 1
                elif cached_speed < 10:
                    gear = 2
                elif cached_speed < 15:
                    gear = 3
                else:
                    gear = 4

            # apply controls immediately so driver can feel it
            car_controls = airsim.CarControls()
            car_controls.steering = float(steer)
            car_controls.throttle = float(throttle)
            car_controls.brake = float(brake)
            car_controls.handbrake = False
            car_controls.is_manual_gear = True
            car_controls.manual_gear = gear
            client.setCarControls(car_controls, vehicle_name='Car1')

            # record at interval
            if now - last_record >= args.recording_interval:
                state, pos, dist, speed = get_state(client, camera_names, goal_pos)
                cached_speed = speed
                # store current state and action
                states.append(state)
                actions.append([steer, throttle, brake])
                last_record = now
                if len(states) % 50 == 0:
                    print(f"Recorded {len(states)} samples, t={elapsed:.1f}s, speed={cached_speed:.2f}")

            # stop conditions
            if args.max_duration and elapsed >= args.max_duration:
                print("Max duration reached")
                break
            time.sleep(args.control_interval)

    except KeyboardInterrupt:
        print("Recording stopped by user")

    finally:
        listener.stop()
        stop_c = airsim.CarControls()
        stop_c.throttle = 0
        stop_c.brake = 1.0
        client.setCarControls(stop_c, vehicle_name='Car1')

    states_np = np.stack(states, axis=0) if states else np.zeros((0,))
    actions_np = np.array(actions, dtype=np.float32)
    return states_np, actions_np


def train_and_save(states: np.ndarray, actions: np.ndarray, args):
    if states.size == 0:
        raise RuntimeError('No recorded states to train on')

    set_seed(args.seed)

    state_dim = states.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeterministicPolicy(state_dim)
    model.to(device)

    ds = StateActionDataset(states, actions)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            preds = model(s_batch)
            loss = loss_fn(preds, a_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * s_batch.size(0)
        avg = total / len(ds)
        print(f"Epoch {epoch+1}/{args.epochs}, loss={avg:.6f}")

    # Save checkpoint in the eval_policy expected format
    ckpt = {
        'policy_state_dict': model.state_dict(),
        'state_dim': state_dim,
        'meta': {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seed': args.seed,
            'epochs': args.epochs,
            'n_samples': len(ds),
        }
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"Saved policy checkpoint to {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/manual_policy/policy_manual.pt')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--cameras', nargs='+', default=['0', 'rear', 'left', 'right'])
    p.add_argument('--goal_x', type=float, default=56.488)
    p.add_argument('--goal_y', type=float, default=3.012)
    p.add_argument('--goal_z', type=float, default=-0.639)
    p.add_argument('--control-interval', type=float, default=0.01)
    p.add_argument('--recording-interval', type=float, default=0.05)
    p.add_argument('--max-duration', type=float, default=None,
                   help='Maximum recording duration in seconds')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    print("Recording state-action pairs...")
    states, actions = record_states(args)
    print(f"Recorded {len(states)} samples")

    print("Training deterministic BC policy...")
    train_and_save(states, actions, args)


if __name__ == '__main__':
    main()
=======
"""Record a reproducible deterministic policy from manual driving.

This script records (state -> action) pairs using the same state construction used
by `examples/eval_policy.py` (image embedding placeholder + scalars), trains a
DeterministicPolicy (MLP backbone + action_head), and saves a checkpoint in the
format `{'policy_state_dict': ..., 'state_dim': ...}` which `eval_policy.py` can load.

Usage:
    python examples/record_and_save_policy.py --out data/manual_policy/policy_manual.pt

Notes:
- Reproducible: sets RNG seeds for torch/numpy/random and disables CuDNN non-determinism.
- State construction: uses the same `image_embedding_placeholder` embedding as `eval_policy.py`.
"""
import os
import time
import argparse
import random
from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import airsim

# ----------------------------- Reproducibility -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic CUDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------- Policy class (match eval_policy) -------------
class DeterministicPolicy(nn.Module):
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
        steering = torch.tanh(actions[:, 0:1])
        throttle = torch.tanh(actions[:, 1:2])
        brake = torch.sigmoid(actions[:, 2:3])
        return torch.cat([steering, throttle, brake], dim=1)

# ----------------------------- State construction helpers -------------------

def image_embedding_placeholder(img: np.ndarray, size=(64, 64)) -> np.ndarray:
    """Same embedding used by eval_policy: resize + RGB flattening.

    Expects an HxWx3 RGB image (uint8). Returns flattened float32 embedding.
    """
    im = cv2.resize(img, size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    return im.ravel()


def get_images(client: airsim.CarClient, camera_names: List[str], vehicle_name='Car1') -> Tuple[np.ndarray, ...]:
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


def get_state(client: airsim.CarClient, camera_names: List[str], goal_pos: np.ndarray, vehicle_name='Car1'):
    imgs = get_images(client, camera_names, vehicle_name)
    embeddings = [image_embedding_placeholder(im, size=(64, 64)) for im in imgs]
    img_emb = np.concatenate(embeddings)

    cs = client.getCarState(vehicle_name=vehicle_name)
    pos = cs.kinematics_estimated.position
    car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])

    vel = cs.kinematics_estimated.linear_velocity
    car_speed = float(np.linalg.norm([vel.x_val, vel.y_val, vel.z_val]))

    dist_to_goal = float(np.linalg.norm(car_pos - goal_pos))

    # placeholder for obstacle distance (keep same default as eval_policy)
    dist_to_obstacle = 10.0

    state = np.concatenate([img_emb, [dist_to_obstacle], [dist_to_goal], [car_speed]])
    return state, car_pos, dist_to_goal, car_speed

# ----------------------------- Dataset & training ---------------------------
class StateActionDataset(Dataset):
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        assert states.shape[0] == actions.shape[0]
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# ----------------------------- Recording + Train workflow -------------------

def record_states(args) -> Tuple[np.ndarray, np.ndarray]:
    print("Connecting to AirSim...")
    client = airsim.CarClient()
    client.confirmConnection()
    print("Connected")

    client.reset()
    time.sleep(0.5)
    client.enableApiControl(True, vehicle_name='Car1')
    try:
        client.armDisarm(True, vehicle_name='Car1')
    except Exception:
        pass

    # release handbrake
    c = airsim.CarControls()
    c.handbrake = False
    client.setCarControls(c, vehicle_name='Car1')
    time.sleep(0.05)

    camera_names = args.cameras
    goal_pos = np.array([args.goal_x, args.goal_y, args.goal_z], dtype=float)

    print("Start driving - press Ctrl+C to stop recording")
    states = []
    actions = []
    start_time = time.time()
    last_record = start_time
    cached_speed = 0.0

    # keyboard control similar to examples/record_trajectory.py
    from pynput import keyboard
    active_keys = set()

    def on_press(key):
        try:
            if hasattr(key, 'char') and key.char:
                active_keys.add(key.char.lower())
            else:
                if key == keyboard.Key.up:
                    active_keys.add('up')
                elif key == keyboard.Key.down:
                    active_keys.add('down')
                elif key == keyboard.Key.left:
                    active_keys.add('left')
                elif key == keyboard.Key.right:
                    active_keys.add('right')
                elif key == keyboard.Key.space:
                    active_keys.add('space')
        except Exception:
            pass

    def on_release(key):
        try:
            if hasattr(key, 'char') and key.char:
                active_keys.discard(key.char.lower())
            else:
                if key == keyboard.Key.up:
                    active_keys.discard('up')
                elif key == keyboard.Key.down:
                    active_keys.discard('down')
                elif key == keyboard.Key.left:
                    active_keys.discard('left')
                elif key == keyboard.Key.right:
                    active_keys.discard('right')
                elif key == keyboard.Key.space:
                    active_keys.discard('space')
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            # compute simple controls from keys
            # steering
            steer = 0.0
            if 'q' in active_keys or 'left' in active_keys:
                steer = -0.5
            if 'd' in active_keys or 'right' in active_keys:
                steer = 0.5
            # throttle/brake
            if ' ' in active_keys:
                throttle = 0.0
                brake = 1.0
            elif 'z' in active_keys or 'up' in active_keys:
                throttle = 1.0
                brake = 0.0
            elif 's' in active_keys or 'down' in active_keys:
                throttle = -1.0
                brake = 0.0
            else:
                throttle = 0.0
                brake = 0.0

            # compute gear (simple)
            if throttle < 0:
                gear = -1
            elif brake > 0.5 or (abs(throttle) < 0.1 and brake > 0.1):
                gear = 0
            else:
                if cached_speed < 5:
                    gear = 1
                elif cached_speed < 10:
                    gear = 2
                elif cached_speed < 15:
                    gear = 3
                else:
                    gear = 4

            # apply controls immediately so driver can feel it
            car_controls = airsim.CarControls()
            car_controls.steering = float(steer)
            car_controls.throttle = float(throttle)
            car_controls.brake = float(brake)
            car_controls.handbrake = False
            car_controls.is_manual_gear = True
            car_controls.manual_gear = gear
            client.setCarControls(car_controls, vehicle_name='Car1')

            # record at interval
            if now - last_record >= args.recording_interval:
                state, pos, dist, speed = get_state(client, camera_names, goal_pos)
                cached_speed = speed
                # store current state and action
                states.append(state)
                actions.append([steer, throttle, brake])
                last_record = now
                if len(states) % 50 == 0:
                    print(f"Recorded {len(states)} samples, t={elapsed:.1f}s, speed={cached_speed:.2f}")

            # stop conditions
            if args.max_duration and elapsed >= args.max_duration:
                print("Max duration reached")
                break
            time.sleep(args.control_interval)

    except KeyboardInterrupt:
        print("Recording stopped by user")

    finally:
        listener.stop()
        stop_c = airsim.CarControls()
        stop_c.throttle = 0
        stop_c.brake = 1.0
        client.setCarControls(stop_c, vehicle_name='Car1')

    states_np = np.stack(states, axis=0) if states else np.zeros((0,))
    actions_np = np.array(actions, dtype=np.float32)
    return states_np, actions_np


def train_and_save(states: np.ndarray, actions: np.ndarray, args):
    if states.size == 0:
        raise RuntimeError('No recorded states to train on')

    set_seed(args.seed)

    state_dim = states.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeterministicPolicy(state_dim)
    model.to(device)

    ds = StateActionDataset(states, actions)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            preds = model(s_batch)
            loss = loss_fn(preds, a_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * s_batch.size(0)
        avg = total / len(ds)
        print(f"Epoch {epoch+1}/{args.epochs}, loss={avg:.6f}")

    # Save checkpoint in the eval_policy expected format
    ckpt = {
        'policy_state_dict': model.state_dict(),
        'state_dim': state_dim,
        'meta': {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seed': args.seed,
            'epochs': args.epochs,
            'n_samples': len(ds),
        }
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"Saved policy checkpoint to {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/manual_policy/policy_manual.pt')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--cameras', nargs='+', default=['0', 'rear', 'left', 'right'])
    p.add_argument('--goal_x', type=float, default=56.488)
    p.add_argument('--goal_y', type=float, default=3.012)
    p.add_argument('--goal_z', type=float, default=-0.639)
    p.add_argument('--control-interval', type=float, default=0.01)
    p.add_argument('--recording-interval', type=float, default=0.05)
    p.add_argument('--max-duration', type=float, default=None,
                   help='Maximum recording duration in seconds')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    print("Recording state-action pairs...")
    states, actions = record_states(args)
    print(f"Recorded {len(states)} samples")

    print("Training deterministic BC policy...")
    train_and_save(states, actions, args)


if __name__ == '__main__':
    main()
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
