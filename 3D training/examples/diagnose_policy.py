<<<<<<< HEAD
"""Quick diagnostic: what is the policy predicting?

This script loads the policy and shows what actions it would predict
for the current state without actually moving the car.
"""
import argparse
import numpy as np
import torch
import airsim
import cv2

from policy import DeterministicPolicy


def image_embedding_placeholder(img: np.ndarray, size=(64, 64)) -> np.ndarray:
    im = cv2.resize(img, size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    return im.ravel()


def get_state(client, camera_names, goal_pos, vehicle_name='Car1'):
    """Get current state."""
    # Get images
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
    
    # Embed images
    embeddings = [image_embedding_placeholder(im, size=(64, 64)) for im in imgs]
    img_emb = np.concatenate(embeddings)
    
    # Get car state
    cs = client.getCarState(vehicle_name=vehicle_name)
    pos = cs.kinematics_estimated.position
    car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
    
    vel = cs.kinematics_estimated.linear_velocity
    car_speed = float(np.linalg.norm([vel.x_val, vel.y_val, vel.z_val]))
    
    dist_to_goal = float(np.linalg.norm(car_pos - goal_pos))
    dist_to_obstacle = 10.0  # Placeholder
    
    state = np.concatenate([img_emb, [dist_to_obstacle], [dist_to_goal], [car_speed]])
    
    return state, car_pos, dist_to_goal, car_speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='data/episodes/policy_bc.pt')
    parser.add_argument('--cameras', nargs='+', default=['0', 'rear', 'left', 'right'])
    parser.add_argument('--goal_x', type=float, default=56.488)
    parser.add_argument('--goal_y', type=float, default=3.012)
    parser.add_argument('--goal_z', type=float, default=-0.639)
    args = parser.parse_args()
    
    # Connect
    print("Connecting to AirSim...")
    client = airsim.CarClient()
    client.confirmConnection()
    print("✓ Connected\n")
    
    # Load policy
    print(f"Loading policy from {args.policy}...")
    state_dim = 4 * 64 * 64 * 3 + 3
    policy = DeterministicPolicy(state_dim)
    policy.load_state_dict(torch.load(args.policy, map_location='cpu'))
    policy.eval()
    print("✓ Policy loaded\n")
    
    goal_pos = np.array([args.goal_x, args.goal_y, args.goal_z])
    
    print("="*80)
    print("POLICY DIAGNOSTIC - Press Ctrl+C to exit")
    print("="*80)
    print("The policy will predict actions based on current car state")
    print("(Car will NOT move - this is just showing predictions)")
    print()
    
    try:
        while True:
            # Get state
            state, car_pos, dist_to_goal, car_speed = get_state(client, args.cameras, goal_pos)
            
            # Predict action
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                action = policy(state_tensor).squeeze(0).cpu().numpy()
            
            steering, throttle, brake = action[0], action[1], action[2]
            
            # Display
            print(f"\rPosition: ({car_pos[0]:6.2f}, {car_pos[1]:6.2f}, {car_pos[2]:6.2f}) | "
                  f"Dist: {dist_to_goal:6.2f}m | "
                  f"Speed: {car_speed:5.2f} | "
                  f"Policy → Steer: {steering:+.3f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}",
                  end='', flush=True)
            
            import time
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\n✓ Done")


if __name__ == '__main__':
    main()
=======
"""Quick diagnostic: what is the policy predicting?

This script loads the policy and shows what actions it would predict
for the current state without actually moving the car.
"""
import argparse
import numpy as np
import torch
import airsim
import cv2

from policy import DeterministicPolicy


def image_embedding_placeholder(img: np.ndarray, size=(64, 64)) -> np.ndarray:
    im = cv2.resize(img, size)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    return im.ravel()


def get_state(client, camera_names, goal_pos, vehicle_name='Car1'):
    """Get current state."""
    # Get images
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
    
    # Embed images
    embeddings = [image_embedding_placeholder(im, size=(64, 64)) for im in imgs]
    img_emb = np.concatenate(embeddings)
    
    # Get car state
    cs = client.getCarState(vehicle_name=vehicle_name)
    pos = cs.kinematics_estimated.position
    car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
    
    vel = cs.kinematics_estimated.linear_velocity
    car_speed = float(np.linalg.norm([vel.x_val, vel.y_val, vel.z_val]))
    
    dist_to_goal = float(np.linalg.norm(car_pos - goal_pos))
    dist_to_obstacle = 10.0  # Placeholder
    
    state = np.concatenate([img_emb, [dist_to_obstacle], [dist_to_goal], [car_speed]])
    
    return state, car_pos, dist_to_goal, car_speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='data/episodes/policy_bc.pt')
    parser.add_argument('--cameras', nargs='+', default=['0', 'rear', 'left', 'right'])
    parser.add_argument('--goal_x', type=float, default=56.488)
    parser.add_argument('--goal_y', type=float, default=3.012)
    parser.add_argument('--goal_z', type=float, default=-0.639)
    args = parser.parse_args()
    
    # Connect
    print("Connecting to AirSim...")
    client = airsim.CarClient()
    client.confirmConnection()
    print("✓ Connected\n")
    
    # Load policy
    print(f"Loading policy from {args.policy}...")
    state_dim = 4 * 64 * 64 * 3 + 3
    policy = DeterministicPolicy(state_dim)
    policy.load_state_dict(torch.load(args.policy, map_location='cpu'))
    policy.eval()
    print("✓ Policy loaded\n")
    
    goal_pos = np.array([args.goal_x, args.goal_y, args.goal_z])
    
    print("="*80)
    print("POLICY DIAGNOSTIC - Press Ctrl+C to exit")
    print("="*80)
    print("The policy will predict actions based on current car state")
    print("(Car will NOT move - this is just showing predictions)")
    print()
    
    try:
        while True:
            # Get state
            state, car_pos, dist_to_goal, car_speed = get_state(client, args.cameras, goal_pos)
            
            # Predict action
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                action = policy(state_tensor).squeeze(0).cpu().numpy()
            
            steering, throttle, brake = action[0], action[1], action[2]
            
            # Display
            print(f"\rPosition: ({car_pos[0]:6.2f}, {car_pos[1]:6.2f}, {car_pos[2]:6.2f}) | "
                  f"Dist: {dist_to_goal:6.2f}m | "
                  f"Speed: {car_speed:5.2f} | "
                  f"Policy → Steer: {steering:+.3f}, Throttle: {throttle:.3f}, Brake: {brake:.3f}",
                  end='', flush=True)
            
            import time
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\n✓ Done")


if __name__ == '__main__':
    main()
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
