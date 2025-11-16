<<<<<<< HEAD
# Parking Car Project - 2D -> 3D

This repository demonstrates a progression from a 2D model-based reinforcement learning parking task (using `highway-env`) to a 3D environment using AirSim and a custom gym-compatible wrapper.


![AirSim Environment](airsim_demo.jpg?raw=true "AirSim Environment")

Purpose
-------
- Provide a compact, reproducible pathway to go from a low-dimensional 2D parking simulation (fast, easy to iterate) to a richer 3D simulation (photorealistic sensors, more realistic dynamics).
- Document the structure, required dependencies, and recommended steps for experimenting and extending the project.

Repository layout
-----------------
`/` - project root

- `2D training/`
  - `parking_model_based.ipynb` : Jupyter notebook implementing a model-based RL pipeline (collect data, learn dynamics, plan with CEM). This is the main 2D demo.
  - `videos/` : recorded example episodes (mp4).

- `3D training/`
  - `dqn_car.py`, `test_dqn.py` and other scripts: examples and training code designed to work with an AirSim environment and the `airgym` wrapper.
  - `airgym/` : gym-like wrappers for AirSim environments (car, drone, etc.).
  - `examples/`, `data/` : utilities, saved trajectories, and example scripts.

High-level workflow
-------------------
1. Iterate quickly in 2D using the notebook. Validate modelling choices and planners on a simple, low-dim state (fast experiments).
2. Transfer insights to 3D: adapt observation representation (images, lidar), adjust action space and dynamics expectations, and use the `airgym` wrappers to interface with AirSim.
3. Use domain randomization, imitation learning, or additional modelling to bridge sim-to-sim and sim-to-real gaps.

Quick start (2D)
----------------
1. Ensure you have Python 3.8+ and a Jupyter environment installed.
2. Install required Python packages (from the notebook):

```bash
pip install highway-env gymnasium moviepy tensorboardx gym pyvirtualdisplay tqdm
```

3. Open the notebook `2D training/parking_model_based.ipynb` in Jupyter or in Colab (adapting paths for Colab if needed). Run cells sequentially. The notebook implements:
   - Random interaction data collection (transitions)
   - A structured dynamics model (LTI-inspired) and training loop
   - A Cross-Entropy Method (CEM) planner to plan actions using the learned model
   - Evaluation and video recording

Notes and tips (2D)
-------------------
- If you see a tqdm ImportError about IProgress when running in plain terminal-based notebooks, install ipywidgets or use the fallback import provided in the notebook.
- The notebook contains a safe `tqdm` import that falls back to text tqdm when ipywidgets is not present.

Quick start (3D / AirSim)
------------------------
1. AirSim requires a running Unreal Engine environment or a prebuilt AirSim binary. Follow AirSim installation instructions:
   - https://microsoft.github.io/AirSim/
2. Install Python dependencies for the `3D training` scripts. Typical requirements (subject to each script):

```bash
pip install gym numpy torch opencv-python airsim scikit-image
# plus any packages listed in the specific example scripts (see `3D training/setup.py` and `requirements.txt` if present)
```

3. Configure `settings.json` in your AirSim project or use the provided `3D training/examples/settings.json` if present. Start the AirSim simulator and ensure you can connect with a small test script (e.g., `3D training/examples/check_distance.py`).
4. Use the `airgym` wrappers to run RL algorithms or testing scripts. For example, to run a DQN baseline, inspect `3D training/dqn_car.py` and `3D training/test_dqn.py`.

Mapping 2D -> 3D (practical checklist)
------------------------------------
- Observations:
  - 2D: low-dimensional vector state (position, heading, velocity).
  - 3D: image frames, lidar scans, or richer state vectors. Consider using pretrained encoders or imitation learning.
- Actions:
  - 2D: continuous low-dim actions (steering, throttle).
  - 3D: similar control variables but expect different scaling/latency.
- Dynamics and delay:
  - 3D dynamics are more complex; include sensor delay, actuator latency, and collision response.
- Training strategy:
  - Validate planners and models in 2D first.
  - When moving to 3D, freeze high-level planner logic and retrain perception / low-level control.
  - Consider behaviour cloning from trajectories collected in the 2D or 3D sim as a warm start.

Troubleshooting
---------------
- ipywidgets / IProgress errors in notebooks: install `ipywidgets` or use the notebook's tqdm fallback.
- AirSim connection issues: ensure the simulator is running and `settings.json` network ports are correct.

Further work
------------
- Add a `requirements.txt` for each folder to pin dependencies.
- Add scripts to automatically launch AirSim and run integration tests.
- Add evaluation scripts to compare planner performance between 2D and 3D.

License and attribution
-----------------------
This project uses `highway-env` and `AirSim`. Respect the licenses of those projects when reusing code or assets.

=======



