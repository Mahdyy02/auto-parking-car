# 3D Training - AirSim + airgym

This folder contains code and examples for running 3D vehicle/drone experiments using Microsoft AirSim and a gym-like wrapper (`airgym`). The goal is to show how to adapt algorithms and workflows from the 2D notebook into a realistic 3D simulator.

Contents overview
-----------------
- `airgym/` : wrapper package that exposes AirSim environments with a gym-compatible API.
- `dqn_car.py`, `test_dqn.py` : example RL scripts for training/testing a DQN agent in a car environment.
- `examples/` : helper scripts and utilities for collecting trajectories, testing manual control, and debugging.
- `data/` : saved episodes and trajectories used for imitation or analysis.

Prerequisites
-------------
1. AirSim: you must have AirSim and a compatible Unreal Engine level or prebuilt AirSim binary. See https://microsoft.github.io/AirSim/ for installation and usage.
2. Python dependencies: typically `airsim`, `gym`, `numpy`, `torch`, `opencv-python`. Install the packages required by the specific scripts you will run.

Example setup steps
-------------------
1. Install Python packages:

```bash
pip install airsim gym numpy torch opencv-python scikit-image
```

2. Launch the AirSim simulator (Unreal Engine) or run a prebuilt environment. Ensure `settings.json` ports and IPs match the client scripts.
3. Test connectivity with a simple script, for example `3D training/examples/check_distance.py`.

Adapting algorithms from 2D to 3D
--------------------------------
- Observation processing: if your 3D agent uses camera frames, add an encoder (CNN) or use pretrained feature extractors. For low-dim state experiments, the AirSim API can return state vectors directly.
- Reward shaping: rewards used in the 2D environment may need re-scaling for the 3D simulator.
- Dynamics differences: tune controllers for the additional inertia and environmental response you see in AirSim.

Data collection and imitation
----------------------------
- Use the `examples/record_and_save_policy.py` or `record_trajectory.py` scripts to produce datasets of states, actions and images. These datasets can initialize policies in the 3D setting via behaviour cloning.

Troubleshooting (common issues)
-------------------------------
- Connection refused: ensure AirSim is running and `settings.json` network config matches the client.
- Unreal crashes or slow rendering: use lower-quality settings or a simpler level for faster iteration.
- Permissions: on some OSes, AirSim requires certain driver or graphics privileges.

Next steps / suggestions
------------------------
- Add a `requirements.txt` for the 3D folder with pins for tested versions.
- Add a small script to convert 2D trajectories to start positions/goals in AirSim for curriculum transfer.
- Integrate domain randomization utilities to improve transfer robustness.

Training videos
---------------
If you recorded training runs (videos) for the 3D experiments, place them in the `videos/` subfolder (create it if missing). Example recommended filename:

```
3D training/videos/training_video.mp4
```

Once the video file is in place you can either open it with your OS default player, or display it inside a Jupyter notebook. A small helper `show_video.py` is provided in this folder to make that easy.

Notes about embedding
---------------------
- Git repositories will often avoid committing large MP4s; consider storing large recordings externally (Google Drive, OneDrive) and linking from this README instead.
- If you want me to embed an existing video into this repository, upload the file (or give me a path I can access) and I will place it into `3D training/videos/` and add an embedded preview to this README.

Embedded preview (training video)
---------------------------------
The repository already contains a recorded training file at `3D training/videos/training.mp4`.
If your browser or Git host supports inline video, the player below will attempt to play it. If it doesn't render in your environment, open the file with your OS video player or use the `show_video.py` helper.

<video controls width="640">
	<source src="videos/training.mp4" type="video/mp4">
	Your browser does not support the video tag. Open `3D training/videos/training.mp4` with a media player to view the recording.
</video>

If you replace the file, keep the same filename (`videos/training.mp4`) or update the `src` path above accordingly.

Notes
-----
AirSim is powerful but heavier to iterate with than the 2D environment. Use the 2D notebook for rapid prototyping and the 3D folder for final validation and perception experiments.
