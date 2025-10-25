"""Check current car position and distance to goal."""
import airsim
import numpy as np

client = airsim.CarClient()
client.confirmConnection()

# Get car state
cs = client.getCarState(vehicle_name='Car1')
pos = cs.kinematics_estimated.position
car_pos = np.array([pos.x_val, pos.y_val, pos.z_val])

# Goal from settings
goal_pos = np.array([56.488, 3.012, -0.639])
goal_threshold = 1.0

# Calculate distance
distance = np.linalg.norm(car_pos - goal_pos)

print("="*80)
print("TASK CONFIGURATION CHECK")
print("="*80)
print(f"Current car position: ({car_pos[0]:.3f}, {car_pos[1]:.3f}, {car_pos[2]:.3f})")
print(f"Goal position:        ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f})")
print(f"Goal threshold:       {goal_threshold} meters")
print()
print(f"3D Distance to goal:  {distance:.3f} meters")
print(f"2D Distance (X,Y):    {np.linalg.norm(car_pos[:2] - goal_pos[:2]):.3f} meters")
print(f"Z difference:         {abs(car_pos[2] - goal_pos[2]):.3f} meters")
print()

if distance < goal_threshold:
    print("❌ PROBLEM: Car is ALREADY at the goal!")
    print(f"   Distance ({distance:.3f}m) < threshold ({goal_threshold}m)")
    print("   Episode will terminate immediately!")
    print()
    print("   FIX: Either move the car away from goal OR change goal position")
elif distance < 5.0:
    print("⚠️  WARNING: Car is very close to goal (< 5m)")
    print("   Task is trivial, episodes will be very short")
elif distance < 10.0:
    print("⚠️  WARNING: Car is quite close to goal (< 10m)")
else:
    print(f"✓ Distance looks good ({distance:.1f}m)")
    print("  Task should require meaningful driving")

print()
print("Recommendation:")
if distance < goal_threshold:
    print("  Move car to starting position at least 20-50 meters from goal")
    print("  OR change goal coordinates to be further away")
