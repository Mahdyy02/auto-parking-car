<<<<<<< HEAD
"""Run a trained BC model in the AirSim environment.

Usage:
    python examples/run_bc_policy.py --model data/bc_models/bc_model.pt
"""
import argparse
import time
import pickle
import os

import numpy as np
import torch
from PIL import Image
import airsim


class BCModelWrapper:
    def __init__(self, model_path, device=None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        data = torch.load(model_path, map_location=self.device)
        # Build model architecture
        from torch import nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(64 * 11 * 11, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3),
                )
            def forward(self, x):
                return self.net(x)
        self.model = Net().to(self.device)
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval()

    def predict_action(self, img):
        # img: HxW or HxWx1 numpy uint8
        import numpy as np
        x = np.array(img, dtype=np.float32) / 255.0
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)  # 1 x C x H x W
        import torch
        t = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            out = self.model(t)
        out = out.cpu().numpy().squeeze().tolist()
        # returns [steering, throttle, brake]
        return float(out[0]), float(out[1]), float(out[2])


def transform_obs_from_airsim(response, out_shape=(84, 84)):
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


def run(args):
    client = airsim.CarClient()
    client.confirmConnection()
    client.reset()
    time.sleep(0.3)
    client.enableApiControl(True, vehicle_name='Car1')
    try:
        client.armDisarm(True, vehicle_name='Car1')
    except Exception:
        pass
    # release handbrake
    c = airsim.CarControls()
    c.handbrake = False
    client.setCarControls(c, vehicle_name='Car1')

    # load model
    model = BCModelWrapper(args.model)
    image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)

    print('Running BC policy. Press Ctrl+C to stop')
    try:
        while True:
            responses = client.simGetImages([image_request])
            img = transform_obs_from_airsim(responses[0], out_shape=(84, 84))

            # get speed for gear
            car_state = client.getCarState(vehicle_name='Car1')
            speed = car_state.speed

            steering, throttle, brake = model.predict_action(img)
            gear = calculate_gear(throttle, speed, brake)

            controls = airsim.CarControls()
            controls.steering = float(steering)
            controls.throttle = float(throttle)
            controls.brake = float(brake)
            controls.handbrake = False
            controls.is_manual_gear = True
            controls.manual_gear = int(gear)

            client.setCarControls(controls, vehicle_name='Car1')
            time.sleep(0.05)

    except KeyboardInterrupt:
        print('\nStopped by user')
        stop_c = airsim.CarControls()
        stop_c.throttle = 0
        stop_c.brake = 1.0
        client.setCarControls(stop_c, vehicle_name='Car1')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='data/bc_models/bc_model.pt')
    args = p.parse_args()
    run(args)
=======
"""Run a trained BC model in the AirSim environment.

Usage:
    python examples/run_bc_policy.py --model data/bc_models/bc_model.pt
"""
import argparse
import time
import pickle
import os

import numpy as np
import torch
from PIL import Image
import airsim


class BCModelWrapper:
    def __init__(self, model_path, device=None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        data = torch.load(model_path, map_location=self.device)
        # Build model architecture
        from torch import nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(64 * 11 * 11, 128),
                    nn.ReLU(),
                    nn.Linear(128, 3),
                )
            def forward(self, x):
                return self.net(x)
        self.model = Net().to(self.device)
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval()

    def predict_action(self, img):
        # img: HxW or HxWx1 numpy uint8
        import numpy as np
        x = np.array(img, dtype=np.float32) / 255.0
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)  # 1 x C x H x W
        import torch
        t = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            out = self.model(t)
        out = out.cpu().numpy().squeeze().tolist()
        # returns [steering, throttle, brake]
        return float(out[0]), float(out[1]), float(out[2])


def transform_obs_from_airsim(response, out_shape=(84, 84)):
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


def run(args):
    client = airsim.CarClient()
    client.confirmConnection()
    client.reset()
    time.sleep(0.3)
    client.enableApiControl(True, vehicle_name='Car1')
    try:
        client.armDisarm(True, vehicle_name='Car1')
    except Exception:
        pass
    # release handbrake
    c = airsim.CarControls()
    c.handbrake = False
    client.setCarControls(c, vehicle_name='Car1')

    # load model
    model = BCModelWrapper(args.model)
    image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)

    print('Running BC policy. Press Ctrl+C to stop')
    try:
        while True:
            responses = client.simGetImages([image_request])
            img = transform_obs_from_airsim(responses[0], out_shape=(84, 84))

            # get speed for gear
            car_state = client.getCarState(vehicle_name='Car1')
            speed = car_state.speed

            steering, throttle, brake = model.predict_action(img)
            gear = calculate_gear(throttle, speed, brake)

            controls = airsim.CarControls()
            controls.steering = float(steering)
            controls.throttle = float(throttle)
            controls.brake = float(brake)
            controls.handbrake = False
            controls.is_manual_gear = True
            controls.manual_gear = int(gear)

            client.setCarControls(controls, vehicle_name='Car1')
            time.sleep(0.05)

    except KeyboardInterrupt:
        print('\nStopped by user')
        stop_c = airsim.CarControls()
        stop_c.throttle = 0
        stop_c.brake = 1.0
        client.setCarControls(stop_c, vehicle_name='Car1')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='data/bc_models/bc_model.pt')
    args = p.parse_args()
    run(args)
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
