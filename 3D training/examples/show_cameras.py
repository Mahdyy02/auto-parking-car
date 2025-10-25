<<<<<<< HEAD
import airsim
import cv2
import numpy as np

client = airsim.CarClient()
client.confirmConnection()

camera_names = ["0", "rear", "left", "right"]

# Example: store last sent control (since AirSim doesn't give it back)
last_steering = 0.0
last_throttle = 0.0
last_brake = 0.0

while True:
    # --- 1. Get camera images ---
    responses = client.simGetImages([
        airsim.ImageRequest(name, airsim.ImageType.Scene, False, False)
        for name in camera_names
    ], vehicle_name="Car1")

    for name, response in zip(camera_names, responses):
        if response.height != 0:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            small = cv2.resize(img_rgb, (320, 180))
            cv2.imshow(f"{name.capitalize()} View", small)

    # --- 2. Get car state ---
    car_state = client.getCarState(vehicle_name="Car1")

    pos = car_state.kinematics_estimated.position
    car_position = (pos.x_val, pos.y_val, pos.z_val)

    vel = car_state.kinematics_estimated.linear_velocity
    car_speed = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])

    orientation = car_state.kinematics_estimated.orientation
    gear = car_state.gear
    handbrake = car_state.handbrake

    print(f"Position: {car_position}, Speed: {car_speed:.2f}, "
          f"Gear: {gear}, Handbrake: {handbrake}, "
          f"Steering: {last_steering}, Throttle: {last_throttle}, Brake: {last_brake}")
    
    # Get collision info
    collision_info = client.simGetCollisionInfo(vehicle_name="Car1")

    if collision_info.has_collided:
        print(f"Collision detected!")
        print(f"Impact point: ({collision_info.impact_point.x_val:.2f}, "
            f"{collision_info.impact_point.y_val:.2f}, "
            f"{collision_info.impact_point.z_val:.2f})")
        print(f"Object hit: {collision_info.object_name}")
        print(f"Penetration depth: {collision_info.penetration_depth:.2f}")
    # else:
    #     print("No collision")
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
=======
import airsim
import cv2
import numpy as np

client = airsim.CarClient()
client.confirmConnection()

camera_names = ["0", "rear", "left", "right"]

# Example: store last sent control (since AirSim doesn't give it back)
last_steering = 0.0
last_throttle = 0.0
last_brake = 0.0

while True:
    # --- 1. Get camera images ---
    responses = client.simGetImages([
        airsim.ImageRequest(name, airsim.ImageType.Scene, False, False)
        for name in camera_names
    ], vehicle_name="Car1")

    for name, response in zip(camera_names, responses):
        if response.height != 0:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            small = cv2.resize(img_rgb, (320, 180))
            cv2.imshow(f"{name.capitalize()} View", small)

    # --- 2. Get car state ---
    car_state = client.getCarState(vehicle_name="Car1")

    pos = car_state.kinematics_estimated.position
    car_position = (pos.x_val, pos.y_val, pos.z_val)

    vel = car_state.kinematics_estimated.linear_velocity
    car_speed = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])

    orientation = car_state.kinematics_estimated.orientation
    gear = car_state.gear
    handbrake = car_state.handbrake

    print(f"Position: {car_position}, Speed: {car_speed:.2f}, "
          f"Gear: {gear}, Handbrake: {handbrake}, "
          f"Steering: {last_steering}, Throttle: {last_throttle}, Brake: {last_brake}")
    
    # Get collision info
    collision_info = client.simGetCollisionInfo(vehicle_name="Car1")

    if collision_info.has_collided:
        print(f"Collision detected!")
        print(f"Impact point: ({collision_info.impact_point.x_val:.2f}, "
            f"{collision_info.impact_point.y_val:.2f}, "
            f"{collision_info.impact_point.z_val:.2f})")
        print(f"Object hit: {collision_info.object_name}")
        print(f"Penetration depth: {collision_info.penetration_depth:.2f}")
    # else:
    #     print("No collision")
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
