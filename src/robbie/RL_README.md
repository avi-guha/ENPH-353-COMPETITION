# Reinforcement Learning Line Following for Robbie

This package implements a Deep Q-Network (DQN) based reinforcement learning system for autonomous line following on the robot in the ENPH-353 competition environment.

## Overview

The system uses:
- **State Representation**: Camera images divided into 20 bins (10 bins in middle third, 10 bins in bottom third)
- **Neural Network**: Deep Q-Network with experience replay
- **Reward System**: Optimized for line following with heavy penalties for collisions
- **Action Space**: 15 discrete actions (3 speeds × 5 steering angles)

## Architecture

### State Space (20 features)
The camera image (800x800) is processed into binary bins:
- **Middle third** (rows 267-533): 10 horizontal bins for lookahead
- **Bottom third** (rows 534-800): 10 horizontal bins for immediate line detection

Each bin indicates presence/absence of the line (white/yellow pixels).

### Action Space (15 actions)
- **Linear velocities**: 0.2, 0.4, 0.6 m/s (slow, medium, fast)
- **Angular velocities**: ±1.5, ±0.8, 0.0 rad/s (hard left/right, left/right, straight)

### Reward Function
The reward system is designed to optimize line following while heavily penalizing collisions:

| Behavior | Reward |
|----------|--------|
| Line in center bins (2-3 active) | +10.0 |
| Line in center bins (1 active) | +5.0 |
| Good lookahead (middle third centered) | +3.0 |
| Forward motion | +2.0 × linear_vel |
| Deviation to left/right | -3.0 |
| Line completely lost | -15.0 |
| Excessive turning | -0.5 × abs(angular_vel) |
| **COLLISION** | **-100.0** (episode ends) |

The discount factor (gamma=0.99) ensures the agent considers future rewards.

## Files

### Core Files
- **`rl_environment.py`**: ROS/Gazebo environment wrapper
  - Handles camera image processing
  - Manages collision detection
  - Implements reward calculation
  - Controls robot movement via `/cmd_vel`
  
- **`dqn_model.py`**: Deep Q-Network implementation
  - Neural network architecture (128-128-64 hidden layers)
  - Experience replay buffer (20,000 capacity)
  - Epsilon-greedy exploration
  - Target network for stable training
  
- **`train_rl.py`**: Main training script
  - Episode management
  - Checkpoint saving/loading
  - Training metrics visualization
  - Periodic model evaluation

- **`run_inference.py`**: Deployment script
  - Load trained models
  - Run inference in real-time
  - Performance evaluation

## Installation

### Prerequisites
```bash
# ROS Noetic with Python 3
# PyTorch
pip install torch torchvision opencv-python matplotlib

# ROS dependencies
sudo apt-get install ros-noetic-cv-bridge ros-noetic-gazebo-ros
```

### Build
```bash
cd ~/ENPH-353-COMPETITION
catkin_make
source devel/setup.bash
```

## Usage

### 1. Start the Simulation

First, launch the Gazebo environment with the robot and collision detection:

```bash
# In terminal 1: Start the simulation
cd ~/ENPH-353-COMPETITION/src/enph353/enph353_utils/scripts
./run_sim.sh -vpgw
```

Make sure the robot spawns correctly. The collision plugin should be active and publishing to `/isHit`.

### 2. Train the Agent

```bash
# In terminal 2: Start training
cd ~/ENPH-353-COMPETITION/src/robbie/scripts
./train_rl.py --episodes 500 --render
```

**Training Options:**
- `--episodes N`: Number of training episodes (default: 500)
- `--max-steps N`: Maximum steps per episode (default: 1000)
- `--save-freq N`: Save checkpoint every N episodes (default: 25)
- `--render`: Display camera view during training
- `--resume PATH`: Resume from checkpoint
- `--checkpoint-dir DIR`: Directory for checkpoints (default: `checkpoints`)

**Example with custom settings:**
```bash
./train_rl.py --episodes 1000 --save-freq 50 --render --checkpoint-dir ./my_models
```

### 3. Monitor Training

Training metrics are automatically saved and plotted:
- `checkpoints/training_metrics.png`: Real-time plots of rewards, losses, epsilon
- `checkpoints/metrics_*.json`: JSON files with detailed metrics
- `checkpoints/dqn_ep_*.pth`: Model checkpoints

The training prints progress every episode:
```
Episode 150/500 | Steps: 823 | Reward: 1847.32 | Avg(100): 1523.45 | Epsilon: 0.123 | Loss: 0.0234 | Collision: No
```

### 4. Run Inference

Once trained, deploy the model:

```bash
# Run for 5 episodes
./run_inference.py checkpoints/dqn_ep_500_*.pth --episodes 5

# Run continuously
./run_inference.py checkpoints/dqn_ep_500_*.pth --continuous

# Run without rendering (faster)
./run_inference.py checkpoints/dqn_ep_500_*.pth --no-render --episodes 10
```

**Inference Options:**
- `checkpoint`: Path to model checkpoint (required)
- `--episodes N`: Number of test episodes (default: 5)
- `--continuous`: Run continuously without resets
- `--no-render`: Disable camera view
- `--no-reset-on-collision`: Continue after collision instead of resetting

### 5. Resume Training

If training is interrupted, resume from the last checkpoint:

```bash
./train_rl.py --resume checkpoints/dqn_ep_interrupted_*.pth --episodes 1000
```

## Training Tips

### Initial Training (Episodes 1-100)
- High exploration (epsilon ~1.0 → 0.6)
- Agent learns basic line detection
- Many collisions expected
- Avg reward: -50 to 100

### Mid Training (Episodes 100-300)
- Moderate exploration (epsilon ~0.6 → 0.2)
- Agent learns steering control
- Fewer collisions
- Avg reward: 100 to 500

### Late Training (Episodes 300+)
- Low exploration (epsilon ~0.2 → 0.05)
- Fine-tuning policy
- Smooth navigation
- Avg reward: 500+

### Signs of Good Training
- ✅ Episode rewards increasing
- ✅ Episode lengths increasing (robot survives longer)
- ✅ Collision rate decreasing
- ✅ Loss stabilizing
- ✅ Epsilon decaying smoothly

### Troubleshooting

**Problem: Agent keeps crashing immediately**
- Solution: Lower initial speed, increase penalty for deviation
- Check that collision detection is working (`rostopic echo /isHit`)

**Problem: Agent won't explore**
- Solution: Increase epsilon_decay to keep exploration longer
- Try curriculum learning: start with easier scenarios

**Problem: Training loss exploding**
- Solution: Lower learning rate (0.0001)
- Check gradient clipping is enabled

**Problem: Agent gets stuck/spins**
- Solution: Add penalty for excessive angular velocity
- Reward forward progress more heavily

## Advanced Configuration

### Modify Reward Function
Edit `rl_environment.py`, function `_calculate_reward()`:
```python
def _calculate_reward(self, state, action_idx):
    # Customize rewards here
    reward = 0.0
    # ... your reward logic
    return reward, done, info
```

### Adjust Network Architecture
Edit `dqn_model.py`, class `DQN`:
```python
self.agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    learning_rate=0.0003,  # Lower for stability
    gamma=0.995,           # Higher for long-term planning
    epsilon_decay=0.997,   # Slower decay for more exploration
    buffer_size=50000,     # Larger buffer for more diverse experiences
    hidden_sizes=[256, 256, 128]  # Larger network
)
```

### Add More Actions
Edit `rl_environment.py`, function `_define_action_space()`:
```python
linear_vels = [0.1, 0.3, 0.5, 0.7]  # More speed options
angular_vels = [2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]  # More steering
```

## Collision Detection

The system uses the existing collision plugin from `enph353_gazebo`. The robot's URDF includes a bumper sensor that detects collisions with:
- Pedestrians
- Other cars (parked and moving)
- Baby Yoda
- Walls
- Any other obstacles

Collisions trigger:
1. Large negative reward (-100)
2. Episode termination
3. Robot reset to starting position

## Performance Expectations

After proper training (500+ episodes):
- **Success Rate**: >80% completion without collision
- **Average Speed**: 0.4-0.6 m/s
- **Smooth Steering**: Low angular velocity variance
- **Line Centering**: Consistently keeps line in center bins

## Monitoring Topics

```bash
# View camera feed
rostopic hz /rrbot/camera1/image_raw

# Monitor velocity commands
rostopic echo /cmd_vel

# Check collision detection
rostopic echo /isHit

# View robot state
rostopic echo /gazebo/get_model_state
```

## Future Improvements

1. **Curriculum Learning**: Start with simplified scenarios, gradually increase difficulty
2. **Prioritized Experience Replay**: Sample important experiences more frequently
3. **Dueling DQN**: Separate value and advantage streams
4. **Multi-Step Returns**: Use n-step TD for better credit assignment
5. **Image Augmentation**: Add noise/blur to camera images for robustness
6. **Transfer Learning**: Pre-train on simpler tracks

## References

- DQN Paper: Mnih et al., "Human-level control through deep reinforcement learning" (2015)
- Double DQN: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2015)
- Experience Replay: Lin, "Self-improving reactive agents based on reinforcement learning" (1992)

## License

This code is part of the ENPH-353 competition package.

## Contact

For issues or questions, please refer to the main competition repository.
