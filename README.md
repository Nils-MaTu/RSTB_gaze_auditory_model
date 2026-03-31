# RSTB Gaze Auditory Model

Code and data for fitting and evaluating a functional opponent-channel
model of gaze-dependent auditory spatial perception.

This repository accompanies the manuscript submission:
**A Functional Model of Gaze-Dependent Spatial Auditory Perception**.

## Directory contents

- `gaze_models.py` - core model, data loaders, fitting, and validation logic.
- `run_fit.py` - entrypoint script that runs fitting/validation and writes results.
- `data/` - input datasets used for model fitting and validation.
- `requirements.txt` - Python dependencies.
- `model_parameters.json` - model parameters written by `run_fit.py`.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 run_fit.py
```
