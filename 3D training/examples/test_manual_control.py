"""Quick test to verify manual control works with keyboard listener.

This script:
1. Sets up keyboard listener (pynput)
2. Sends controls to AirSim via API
3. Prints what you're pressing and car movement
4. Verifies the car actually moves

Run this and press WASD/Arrow keys.
You should see the controls and position change.
"""
import airsim
import time
import numpy as np

try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    print("ERROR: pynput not installed!")
    print("Install with: pip install pynput")
    exit(1)


class SimpleKeyboardController:
    """Simple keyboard controller."""
    def __init__(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.active_keys = set()
        
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
    
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
        if 'q' in self.active_keys:
            self.steering = -0.5
        elif 'd' in self.active_keys:
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
        self.listener.stop()

print("Connecting to AirSim...")
client = airsim.CarClient()
client.confirmConnection()
print("✓ Connected\n")

# Setup keyboard controller
print("Setting up keyboard controller...")
controller = SimpleKeyboardController()
time.sleep(0.5)
print("✓ Keyboard ready\n")

# Enable API control
client.enableApiControl(True, vehicle_name='Car1')
try:
    client.armDisarm(True, vehicle_name='Car1')
except:
    pass

print("="*80)
print("MANUAL CONTROL TEST")
print("="*80)
print("Controls: Z/↑=Forward, S/↓=Brake, Q/←=Left, D/→=Right, SPACE=Emergency Brake")
print("This script will show what controls are being applied")
print("Press Ctrl+C to stop\n")

try:
    for i in range(100):  # Run for 100 steps (20 seconds)
        # Get keyboard controls
        steering, throttle, brake = controller.get_controls()
        
        # Get car state FIRST (before applying controls) to get current speed
        cs = client.getCarState(vehicle_name='Car1')
        pos = cs.kinematics_estimated.position
        vel = cs.kinematics_estimated.linear_velocity
        speed = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])
        
        # Apply to car
        controls = airsim.CarControls()
        controls.steering = steering
        controls.throttle = throttle
        controls.brake = brake
        
        # Automatic gear shifting based on speed and throttle direction
        if throttle < 0:
            # Reverse gear
            controls.is_manual_gear = True
            controls.manual_gear = -1
        elif throttle == 0 or brake > 0:
            # Neutral when coasting or braking
            controls.is_manual_gear = True
            controls.manual_gear = 0
        else:
            # Forward gears: shift based on speed (m/s)
            # Gear 1: 0-5 m/s, Gear 2: 5-10 m/s, Gear 3: 10-15 m/s, Gear 4: 15+ m/s
            controls.is_manual_gear = True
            if speed < 5:
                controls.manual_gear = 1
            elif speed < 10:
                controls.manual_gear = 2
            elif speed < 15:
                controls.manual_gear = 3
            else:
                controls.manual_gear = 4
        
        client.setCarControls(controls, vehicle_name='Car1')
        
        # Show gear as R (reverse), N (neutral), or 1-4 (forward)
        gear_display = 'R' if controls.manual_gear == -1 else ('N' if controls.manual_gear == 0 else str(controls.manual_gear))
        
        print(f"Step {i+1:3d}: "
              f"steer={steering:+.3f}, throttle={throttle:+.3f}, brake={brake:.3f} | "
              f"gear={gear_display} | "
              f"pos=({pos.x_val:7.2f}, {pos.y_val:7.2f}, {pos.z_val:6.2f}) | "
              f"speed={speed:5.2f} m/s")
        
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n\nTest stopped\n")

# Cleanup
controller.stop()

# Stop the car
stop_controls = airsim.CarControls()
stop_controls.throttle = 0
stop_controls.brake = 1.0
client.setCarControls(stop_controls, vehicle_name='Car1')

client.enableApiControl(False, vehicle_name='Car1')

print("="*80)
print("RESULTS:")
print("="*80)
print("If you saw:")
print("  ✓ Controls changing when you pressed keys → GOOD!")
print("  ✓ Position changing → GOOD!")
print("  ✓ Speed > 0 when pressing throttle → GOOD!")
print()
print("If you saw:")
print("  ✗ Controls always 0.000 → Problem with keyboard input")
print("  ✗ Position not changing → Car is frozen/not responding")
print("  ✗ Speed always 0 → Car engine not working")
