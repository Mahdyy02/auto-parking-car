"""Keyboard runner for VirtualController

Run this script from the repository root (cmd.exe):

    python examples\keyboard_control.py

Make sure AirSim is running before you start. Press Esc to stop the controller,
or Ctrl-C to interrupt the script.
"""

import time

try:
    from simple_airsim.api.virtual_controller import VirtualController
except Exception as e:
    print('VirtualController not available:', e)
    raise


def main():
    vc = VirtualController()
    with vc:
        vc.enable()
        print('Virtual controller enabled. Use ZQSD/arrow keys, R/F, Q/E, T/L, Space, Esc to control.')
        try:
            # Keep the script alive while controller is enabled. The controller
            # will disable itself when Esc is pressed.
            while getattr(vc, '_enabled', False):
                time.sleep(0.1)
        except KeyboardInterrupt:
            print('Interrupted by user; disabling controller...')
            vc.disable()


if __name__ == '__main__':
    main()
