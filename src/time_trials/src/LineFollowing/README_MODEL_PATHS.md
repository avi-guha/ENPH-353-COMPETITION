# Model File Paths - Fixed

## Problem
The training script and inference node were using relative paths, which could resolve to different locations depending on where the scripts were run from.

## Solution
Both scripts now use **absolute paths** based on the script's directory location.

## File Locations

**Model file will be saved/loaded at:**
```
/home/fizzer/ENPH-353-COMPETITION/src/time_trials/src/LineFollowing/best_model.pth
```

## Usage

### Training
```bash
# Run from anywhere - model saves to the correct location
python3 /home/fizzer/ENPH-353-COMPETITION/src/time_trials/src/LineFollowing/train_model.py
```

The script will print the full path where the model is saved.

### Inference
```bash
# Run the inference node - it will look for the model in the correct location
rosrun time_trials inference_node.py
```

Or with a custom model path:
```bash
rosrun time_trials inference_node.py _model_path:=/path/to/your/model.pth
```

## Verification

After training completes, verify the model exists:
```bash
ls -lh /home/fizzer/ENPH-353-COMPETITION/src/time_trials/src/LineFollowing/best_model.pth
```

If the inference node still can't find it, check the logs - it will print the exact path it's looking for.
