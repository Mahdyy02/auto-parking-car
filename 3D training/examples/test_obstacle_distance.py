"""Quick test script to verify geometric obstacle distance calculation.

Run this while AirSim is running to see real-time distance readings.
Press Ctrl-C to stop.
"""

import time
import airsim
import sys
import os

# Add parent directory to path to import from collect_episodes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collect_episodes import nearest_obstacle_distance_geometric

def main():
    print("Connecting to AirSim...")
    client = airsim.CarClient()
    client.confirmConnection()
    print("Connected!")
    
    # First, list all scene objects to help identify obstacles
    print("\n" + "="*70)
    print("SCENE OBJECT DISCOVERY")
    print("="*70)
    try:
        all_objects = client.simListSceneObjects()
        print(f"Found {len(all_objects)} objects in scene:")
        for i, obj in enumerate(all_objects[:20]):  # Show first 20
            print(f"  {i+1}. {obj}")
        if len(all_objects) > 20:
            print(f"  ... and {len(all_objects) - 20} more")
    except Exception as e:
        print(f"Could not list scene objects: {e}")
    
    print("\n" + "="*70)
    print("REAL-TIME DISTANCE MONITORING")
    print("="*70)
    print("Drive your car around manually in the simulator to see distances change.")
    print("Press Ctrl-C to stop.\n")
    
    last_collision_object = None
    collision_count = 0
    
    try:
        while True:
            # Get collision info FIRST (before distance calculation)
            coll = client.simGetCollisionInfo(vehicle_name='Car1')
            
            # Get distance
            dist = nearest_obstacle_distance_geometric(
                client,
                car_half_length=2.6,
                car_half_width=1.05,
                obstacle_objects=None  # auto-detect
            )
            
            # Get car state for additional context
            state = client.getCarState(vehicle_name='Car1')
            pos = state.kinematics_estimated.position
            speed = state.speed
            
            # Print status with detailed collision info
            if coll.has_collided:
                collision_count += 1
                status = "🔴 COLLISION!"
                
                # Show collision warning with object name and distance
                collision_details = (
                    f"\n{'='*70}\n"
                    f"⚠️  COLLISION DETECTED #{collision_count}\n"
                    f"{'='*70}\n"
                    f"  Object Name: '{coll.object_name}'\n"
                    f"  Collision Point: ({coll.impact_point.x_val:.2f}, {coll.impact_point.y_val:.2f}, {coll.impact_point.z_val:.2f})\n"
                    f"  Penetration Depth: {coll.penetration_depth:.3f}m\n"
                    f"  Time Since Start: {coll.time_stamp / 1e9:.2f}s\n"
                    f"  Measured Distance: {dist:.3f}m\n"
                    f"{'='*70}"
                )
                
                if coll.object_name != last_collision_object:
                    print(collision_details)
                    last_collision_object = coll.object_name
                else:
                    # Compact format for continued collision
                    print(f"{status} | Distance: {dist:6.3f}m | Object: '{coll.object_name}' | "
                          f"Penetration: {coll.penetration_depth:.3f}m | Speed: {speed:5.2f}m/s")
            else:
                status = "✅ Clear"
                last_collision_object = None
                print(f"{status} | Obstacle distance: {dist:6.2f}m | "
                      f"Speed: {speed:5.2f}m/s | "
                      f"Pos: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})")
            
            # time.sleep(0.2)
            
    except KeyboardInterrupt:
        print("\n\nStopped.")

if __name__ == '__main__':
    main()
