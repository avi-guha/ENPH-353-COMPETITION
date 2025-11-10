# 🚀 RL Line Following - Quick Reference Card

## One-Line Quick Start
```bash
cd ~/ENPH-353-COMPETITION/src/robbie/scripts && ./quickstart.py train --episodes 500 --render
```

## Essential Commands

### Setup (Once)
```bash
pip install -r requirements.txt
cd ~/ENPH-353-COMPETITION && catkin_make && source devel/setup.bash
```

### Training
```bash
./quickstart.py train --episodes 500              # Basic training
./quickstart.py train --episodes 1000 --render    # With visualization
./quickstart.py train --resume checkpoints/*.pth  # Resume training
```

### Testing
```bash
./quickstart.py test                              # Test latest model
./quickstart.py test --checkpoint path/to/model   # Test specific model
./quickstart.py test --continuous                 # Run continuously
```

### Debugging
```bash
./quickstart.py info                              # System status
./quickstart.py visualize                         # Camera bins viewer
```

## File Locations

| File | Purpose |
|------|---------|
| `rl_environment.py` | Environment & rewards |
| `dqn_model.py` | Neural network |
| `train_rl.py` | Training loop |
| `run_inference.py` | Testing/deployment |
| `config.py` | All hyperparameters |
| `checkpoints/` | Saved models |

## Quick Tuning

### Problem: Too many crashes
```python
# Edit config.py
'linear_velocities': [0.15, 0.3, 0.45]  # Slower
'collision_penalty': -150.0              # Harsher penalty
```

### Problem: Too slow/cautious
```python
'linear_velocities': [0.3, 0.5, 0.7]    # Faster
'forward_motion_scale': 3.0             # Reward speed more
```

### Problem: Won't learn
```python
'learning_rate': 0.001                   # Increase
'epsilon_decay': 0.997                   # Explore longer
```

## Key Metrics to Watch

| Metric | Good | Bad |
|--------|------|-----|
| Episode Reward | Increasing | Flat/decreasing |
| Collision Rate | <10% | >30% |
| Episode Length | Increasing | Stuck low |
| Epsilon | Decaying smoothly | Stuck at 1.0 |

## ROS Topics

```bash
rostopic echo /rrbot/camera1/image_raw   # Camera feed
rostopic echo /cmd_vel                   # Robot commands
rostopic echo /isHit                     # Collision detection
```

## Training Timeline

| Episodes | Epsilon | Behavior | Avg Reward |
|----------|---------|----------|------------|
| 1-100 | 1.0→0.6 | Random, crashes | -50 to 100 |
| 100-300 | 0.6→0.2 | Learning steering | 100 to 500 |
| 300-500 | 0.2→0.05 | Refined policy | 500+ |

## Success Criteria
- ✅ 80%+ episodes without collision
- ✅ Average speed 0.4-0.5 m/s
- ✅ Line stays in center bins
- ✅ Smooth steering (low angular velocity variance)

## Emergency Commands
```bash
# Kill everything
pkill -f gzserver; pkill -f gzclient; killall -9 rosout roslaunch rosmaster

# Check ROS
rostopic list

# Check GPU
nvidia-smi    # If using CUDA
```

## Documentation
- Full guide: `RL_README.md`
- Project summary: `PROJECT_SUMMARY.md`
- Config options: `config.py`

---
**Need help?** Run `./quickstart.py info` to check system status
