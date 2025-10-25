"""VirtualController

Listens for keyboard events and sends simple API calls to AirSim to control
the drone. Designed to be small and self-contained so `GUIManager` can use
it without extra wiring.

Default key mapping:
  Z / UpArrow    - forward
  S / DownArrow  - backward
  Q / LeftArrow  - left
  D / RightArrow - right
  R              - ascend
  F              - descend
  Q              - yaw left
  E              - yaw right
  T              - takeoff
  L              - land
  Space          - hover
  Esc            - disable controller / stop listener

Notes:
- Uses pynput for keyboard capture (requirements.txt already lists pynput).
- Creates its own airsim.MultirotorClient() and enables API control on enable().
- Commands use short-duration velocity or yaw-rate commands so they feel
  responsive to key presses.
"""

import threading
import time
import traceback

import airsim
from pynput import keyboard


class VirtualController:
    def __init__(self, client: airsim.MultirotorClient = None, speed: float = 5.0,
                 z_speed: float = 3.0, yaw_rate: float = 60.0, duration: float = 0.5):
        """
        :param client: Optional AirSim client. If omitted, a new MultirotorClient
                       will be created and connected.
        :param speed: Horizontal speed in m/s for WASD/arrow movement.
        :param z_speed: Vertical speed in m/s for ascend/descend.
        :param yaw_rate: Yaw rate in degrees/sec for Q/E.
        :param duration: Command duration (seconds) for velocity/yaw commands.
        """
        self.client = client
        self._owns_client = client is None
        self.speed = speed
        self.z_speed = z_speed
        self.yaw_rate = yaw_rate
        self.duration = duration

        self._listener = None
        self._thread = None
        self._enabled = False
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disable()
        # don't suppress exceptions
        return False

    def enable(self):
        """Start listening and enable API control on the AirSim client."""
        with self._lock:
            if self._enabled:
                return
            try:
                if self.client is None:
                    self.client = airsim.MultirotorClient()
                    # best-effort connection
                    try:
                        self.client.confirmConnection()
                    except Exception:
                        # continue even if confirm fails; handle errors later
                        pass

                try:
                    self.client.enableApiControl(True)
                except Exception as e:
                    # if enabling fails, print and continue: commands will likely fail later
                    print('VirtualController: enableApiControl(True) failed:', e)
                    # Don't print full traceback to avoid spam, but continue

                self._enabled = True
                # listener runs in background thread
                self._thread = threading.Thread(target=self._start_listener, daemon=True)
                self._thread.start()
                print('VirtualController enabled: keys control the drone (W/A/S/D, R/F, Q/E, T/L, Space, Esc)')
                print('Note: If Unreal crashes, make sure AirSim is fully loaded before starting this script.')
            except Exception:
                print('VirtualController enable failed:')
                traceback.print_exc()

    def disable(self):
        """Stop listening and (optionally) disable API control."""
        with self._lock:
            if not self._enabled:
                return
            try:
                if self._listener is not None:
                    try:
                        self._listener.stop()
                    except Exception:
                        pass
                self._enabled = False
                # optionally hover when disabling
                try:
                    self.client.hoverAsync().join(timeout=1)
                except Exception:
                    pass
                try:
                    # leave API control enabled or call disable if you want
                    # self.client.enableApiControl(False)
                    pass
                except Exception:
                    pass
                print('VirtualController disabled')
            except Exception:
                print('Error disabling VirtualController:')
                traceback.print_exc()

    # ---- internal listener ----
    def _start_listener(self):
        try:
            with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
                self._listener = listener
                listener.join()
        except Exception:
            print('VirtualController: keyboard listener failed:')
            traceback.print_exc()

    def _on_press(self, key):
        # Map pressed key to an action. Non-blocking.
        try:
            # printable keys
            if hasattr(key, 'char') and key.char is not None:
                c = key.char.lower()
                if c == 'z':
                    self._move_forward()
                elif c == 's':
                    self._move_backward()
                elif c == 'q':
                    self._move_left()
                elif c == 'd':
                    self._move_right()
                elif c == 't':
                    self._takeoff()
                elif c == 'l':
                    self._land()
                elif c == ' ':
                    self._hover()
            else:
                # special keys
                if key == keyboard.Key.up:
                    self._ascend()
                elif key == keyboard.Key.down:
                    self._descend()
                elif key == keyboard.Key.left:
                    self._yaw_left()
                elif key == keyboard.Key.right:
                    self._yaw_right()
                elif key == keyboard.Key.esc:
                    # ESC disables controller
                    print('VirtualController: ESC pressed, disabling controller')
                    self.disable()
        except Exception:
            print('Error handling key press:')
            traceback.print_exc()

    def _on_release(self, key):
        # We don't use key release in this simple controller.
        return

    # ---- action methods ----
    def _send_velocity(self, vx: float, vy: float, vz: float):
        try:
            # Non-blocking; short duration to make control reactive
            self.client.moveByVelocityAsync(vx, vy, vz, self.duration)
        except Exception:
            print('Failed to send velocity command:')
            traceback.print_exc()

    def _send_yaw_rate(self, rate_deg: float):
        try:
            # Try AirSim rotateByYawRateAsync first
            if hasattr(self.client, 'rotateByYawRateAsync'):
                try:
                    self.client.rotateByYawRateAsync(rate_deg, self.duration)
                    return
                except Exception:
                    # fallthrough to using yaw mode
                    pass

            # Fall back to moveByVelocity with yaw_mode controlling yaw rate
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=rate_deg)
            # zero linear velocity, keep current z (approx 0), small duration
            try:
                # Use moveByVelocityAsync with yaw control
                self.client.moveByVelocityAsync(0, 0, 0, self.duration, yaw_mode=yaw_mode)
            except Exception:
                # As last resort, use client.rotateToYawAsync (may be blocking)
                try:
                    current = self.client.getMultirotorState().kinematics_estimated.orientation
                    # best-effort, do nothing else
                except Exception:
                    pass
        except Exception:
            print('Failed to send yaw rate:')
            traceback.print_exc()

    def _takeoff(self):
        try:
            self.client.takeoffAsync().join()
        except Exception:
            print('Takeoff failed:')
            traceback.print_exc()

    def _land(self):
        try:
            self.client.landAsync().join()
        except Exception:
            print('Land failed:')
            traceback.print_exc()

    def _hover(self):
        try:
            self.client.hoverAsync().join(timeout=1)
        except Exception:
            print('Hover failed:')
            traceback.print_exc()

    def _move_forward(self):
        self._send_velocity(self.speed, 0, 0)

    def _move_backward(self):
        self._send_velocity(-self.speed, 0, 0)

    def _move_left(self):
        self._send_velocity(0, -self.speed, 0)

    def _move_right(self):
        self._send_velocity(0, self.speed, 0)

    def _ascend(self):
        # Up in NED is negative z velocity
        self._send_velocity(0, 0, -self.z_speed)

    def _descend(self):
        self._send_velocity(0, 0, self.z_speed)

    def _yaw_left(self):
        self._send_yaw_rate(-self.yaw_rate)

    def _yaw_right(self):
        self._send_yaw_rate(self.yaw_rate)
