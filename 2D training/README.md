<<<<<<< HEAD
# 2D Training - Parking (highway-env)

This folder contains the 2D model-based reinforcement learning demo using `highway-env`.

Contents
--------
- `parking_model_based.ipynb` : The central Jupyter notebook demonstrating the pipeline.
- `videos/` : Example recorded episodes.

What this demo shows
--------------------
1. Random data collection from the `parking-v0` environment.
2. A structured dynamics model (linearization-like) trained by supervised learning on collected transitions.
3. A Cross-Entropy Method (CEM) planner that uses the learned dynamics for planning vehicle trajectories.
4. Evaluation and recorded videos.

How to run
----------
1. Install dependencies (run in terminal):

```bash
pip install highway-env gymnasium moviepy tensorboardx gym pyvirtualdisplay tqdm
```

2. Launch Jupyter and open `parking_model_based.ipynb`.
3. Execute cells in order. Note that some notebook cells use shell `pip` commands; if your environment already has the packages installed, you can skip those cells.

Notes
-----
- The notebook includes a safe import for `tqdm` that falls back when notebook widgets are not installed. If you want the rich progress bars, install `ipywidgets`.
- Use the 2D experiments to tune hyperparameters (model architecture, planner horizon, population size) before moving to 3D.

Next steps
----------
- Export successful policies or trajectories and use them as demonstrations for the 3D training pipeline (behaviour cloning or curriculum learning).
=======
# 2D Training - Parking (highway-env)

This folder contains the 2D model-based reinforcement learning demo using `highway-env`.

Contents
--------
- `parking_model_based.ipynb` : The central Jupyter notebook demonstrating the pipeline.
- `videos/` : Example recorded episodes.

What this demo shows
--------------------
1. Random data collection from the `parking-v0` environment.
2. A structured dynamics model (linearization-like) trained by supervised learning on collected transitions.
3. A Cross-Entropy Method (CEM) planner that uses the learned dynamics for planning vehicle trajectories.
4. Evaluation and recorded videos.

How to run
----------
1. Install dependencies (run in terminal):

```bash
pip install highway-env gymnasium moviepy tensorboardx gym pyvirtualdisplay tqdm
```

2. Launch Jupyter and open `parking_model_based.ipynb`.
3. Execute cells in order. Note that some notebook cells use shell `pip` commands; if your environment already has the packages installed, you can skip those cells.

Notes
-----
- The notebook includes a safe import for `tqdm` that falls back when notebook widgets are not installed. If you want the rich progress bars, install `ipywidgets`.
- Use the 2D experiments to tune hyperparameters (model architecture, planner horizon, population size) before moving to 3D.

Next steps
----------
- Export successful policies or trajectories and use them as demonstrations for the 3D training pipeline (behaviour cloning or curriculum learning).
>>>>>>> 01cdaa58d9b2812ef465bed3c21fe5ecb0cc57fb
