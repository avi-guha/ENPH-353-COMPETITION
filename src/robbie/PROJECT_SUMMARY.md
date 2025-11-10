# Reinforcement Learning Line Following - Project Summary

## 🎯 Project Overview

This is a complete **Deep Q-Network (DQN) based reinforcement learning system** for autonomous line following on a robot in the ENPH-353 Gazebo simulation environment. The system uses camera vision divided into bins as state representation and learns optimal steering/speed control through trial and error with heavy penalties for collisions.

---

## 📋 What Was Created

### Core Components

1. **`rl_environment.py`** (431 lines)
   - Complete ROS/Gazebo environment wrapper
   - Camera image processing into 20 binary bins
   - Reward function with collision detection
   - Robot control via `/cmd_vel`
   - Episode management and reset functionality

2. **`dqn_model.py`** (334 lines)
   - Deep Q-Network neural architecture
   - Experience replay buffer (20,000 capacity)
   - Epsilon-greedy exploration strategy
   - Target network for stable training
   - Checkpoint save/load functionality

3. **`train_rl.py`** (274 lines)
   - Main training loop with episode management
   - Automatic checkpoint saving every 25 episodes
   - Real-time metrics plotting (rewards, loss, epsilon)
   - Resume training from checkpoints
   - Comprehensive logging

4. **`run_inference.py`** (186 lines)
   - Load trained models for deployment
   - Run episodes or continuous inference
   - Performance evaluation metrics
   - Real-time visualization option

5. **`visualize_bins.py`** (246 lines)
   - Debug tool for camera bin visualization
   - Shows how images are processed into state
   - Real-time display with overlays
   - Position estimation feedback

6. **`quickstart.py`** (239 lines)
   - Unified interface for all operations
   - Dependency checking
   - System information display
   - Simplified command-line usage

### Supporting Files

- **`RL_README.md`**: Comprehensive documentation (350+ lines)
- **`requirements.txt`**: Python dependencies
- **`rl_training.launch`**: ROS launch file for setup
- **Modified `robbie.xacro`**: Added collision detection sensor

---

## 🧠 Technical Design

### State Representation (20 features)
```
Camera Image (800x800 pixels)
├── Middle Third (rows 267-533)
│   └── 10 horizontal bins → lookahead for steering
└── Bottom Third (rows 534-800)
    └── 10 horizontal bins → immediate line detection

Each bin = binary (0/1) indicating white/yellow line presence
```

### Action Space (15 discrete actions)
```
Linear Velocity × Angular Velocity:
├── Slow (0.2 m/s)    × {-1.5, -0.8, 0.0, +0.8, +1.5} rad/s
├── Medium (0.4 m/s)  × {-1.5, -0.8, 0.0, +0.8, +1.5} rad/s
└── Fast (0.6 m/s)    × {-1.5, -0.8, 0.0, +0.8, +1.5} rad/s
```

### Reward Structure
| Condition | Reward | Purpose |
|-----------|--------|---------|
| Line in center bins (2-3) | +10.0 | Strong centering incentive |
| Line in center bins (1) | +5.0 | Moderate centering |
| Good lookahead | +3.0 | Anticipatory steering |
| Forward motion | +2.0 × speed | Progress reward |
| Deviation left/right | -3.0 | Keep centered |
| Line lost | -15.0 | Stay on track |
| Excessive turning | -0.5 × |ω| | Smooth driving |
| **COLLISION** | **-100.0** | **Critical penalty** |

**Discount Factor (γ)**: 0.99 for long-term planning

### Neural Network Architecture
```
Input Layer:    20 features (bin states)
Hidden Layer 1: 128 neurons + ReLU + Dropout(0.2)
Hidden Layer 2: 128 neurons + ReLU + Dropout(0.2)
Hidden Layer 3: 64 neurons + ReLU + Dropout(0.2)
Output Layer:   15 Q-values (one per action)

Optimizer: Adam (lr=0.0005)
Loss: Smooth L1 (Huber Loss)
Parameters: ~19,000
```

### Training Algorithm: DQN with Experience Replay
```python
1. Initialize Q-network and target network
2. For each episode:
   a. Reset environment
   b. For each step:
      - Select action (ε-greedy)
      - Execute action, observe reward and next state
      - Store transition in replay buffer
      - Sample random batch from buffer
      - Compute TD target using target network
      - Update Q-network via gradient descent
   c. Decay ε (exploration rate)
   d. Periodically update target network (every 10 episodes)
   e. Save checkpoint (every 25 episodes)
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
# Install Python dependencies
cd ~/ENPH-353-COMPETITION/src/robbie
pip install -r requirements.txt

# Build ROS workspace
cd ~/ENPH-353-COMPETITION
catkin_make
source devel/setup.bash
```

### 2. Start Simulation
```bash
# Terminal 1: Launch Gazebo with robot
cd ~/ENPH-353-COMPETITION/src/enph353/enph353_utils/scripts
./run_sim.sh -vpgw
```

### 3. Train the Agent
```bash
# Terminal 2: Start training
cd ~/ENPH-353-COMPETITION/src/robbie/scripts
./quickstart.py train --episodes 500 --render
```

### 4. Test the Agent
```bash
# After training completes or from another terminal
./quickstart.py test --episodes 5
```

### 5. Debug/Visualize
```bash
# See how camera bins work in real-time
./quickstart.py visualize
```

---

## 📊 Expected Training Progress

### Phase 1: Exploration (Episodes 1-100)
- **Epsilon**: 1.0 → 0.6
- **Behavior**: Random, many collisions
- **Avg Reward**: -50 to 100
- **Key Learning**: Line detection basics

### Phase 2: Skill Development (Episodes 100-300)
- **Epsilon**: 0.6 → 0.2
- **Behavior**: Basic steering, fewer collisions
- **Avg Reward**: 100 to 500
- **Key Learning**: Centering and collision avoidance

### Phase 3: Refinement (Episodes 300-500)
- **Epsilon**: 0.2 → 0.05
- **Behavior**: Smooth navigation, rare collisions
- **Avg Reward**: 500+
- **Key Learning**: Optimal policy convergence

### Success Metrics
- ✅ **Completion Rate**: >80% without collision
- ✅ **Average Speed**: 0.4-0.6 m/s
- ✅ **Line Centering**: Consistently in center bins
- ✅ **Smooth Steering**: Low angular velocity variance

---

## 🎮 Usage Examples

### Training Commands
```bash
# Basic training
./quickstart.py train --episodes 500

# Training with visualization
./quickstart.py train --episodes 1000 --render

# Resume from checkpoint
./quickstart.py train --resume checkpoints/dqn_ep_interrupted_*.pth

# Custom checkpoint directory
./quickstart.py train --episodes 500 --checkpoint-dir my_models
```

### Inference Commands
```bash
# Test latest model (5 episodes)
./quickstart.py test

# Test specific checkpoint (10 episodes)
./quickstart.py test --checkpoint checkpoints/dqn_ep_500_*.pth --episodes 10

# Continuous driving (no resets)
./quickstart.py test --continuous

# Headless testing (no visualization)
./quickstart.py test --no-render --episodes 20
```

### Utility Commands
```bash
# Check system status
./quickstart.py info

# Visualize camera processing
./quickstart.py visualize
```

---

## 📁 Project Structure

```
src/robbie/
├── scripts/
│   ├── rl_environment.py      # Environment wrapper
│   ├── dqn_model.py           # Neural network & agent
│   ├── train_rl.py            # Training script
│   ├── run_inference.py       # Inference script
│   ├── visualize_bins.py      # Debug visualizer
│   └── quickstart.py          # Unified interface
├── urdf/
│   └── robbie.xacro           # Robot URDF (modified with collision sensor)
├── launch/
│   └── rl_training.launch     # ROS launch file
├── requirements.txt           # Python dependencies
├── RL_README.md              # Full documentation
└── checkpoints/              # Created during training
    ├── dqn_ep_*.pth          # Model checkpoints
    ├── metrics_*.json        # Training metrics
    └── training_metrics.png  # Visualization plots
```

---

## 🔧 Advanced Customization

### Modify Reward Function
Edit `rl_environment.py`, method `_calculate_reward()`:
```python
# Example: Add penalty for slow speed
if linear_vel < 0.3:
    reward -= 2.0  # Encourage faster driving
```

### Adjust Network Size
Edit `train_rl.py`, agent initialization:
```python
self.agent = DQNAgent(
    hidden_sizes=[256, 256, 128],  # Larger network
    learning_rate=0.0003,
    buffer_size=50000
)
```

### Add More Actions
Edit `rl_environment.py`, method `_define_action_space()`:
```python
linear_vels = [0.1, 0.3, 0.5, 0.7]  # 4 speeds
angular_vels = [2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]  # 7 angles
# Creates 28 total actions
```

---

## 🐛 Troubleshooting

### Issue: Robot crashes immediately
**Solution**: Lower initial speeds, increase deviation penalty
```python
linear_vels = [0.15, 0.3, 0.45]  # Slower speeds
```

### Issue: Training loss exploding
**Solution**: Reduce learning rate
```python
learning_rate=0.0001  # Lower from default 0.0005
```

### Issue: Agent won't explore
**Solution**: Slower epsilon decay
```python
epsilon_decay=0.997  # Slower decay keeps exploration longer
```

### Issue: No collision detection
**Solution**: Check collision plugin
```bash
rostopic echo /isHit  # Should show True when colliding
```

---

## 📈 Monitoring During Training

### Terminal Output
```
Episode 245/500 | Steps: 876 | Reward: 2134.56 | Avg(100): 1823.45 | 
Epsilon: 0.134 | Loss: 0.0187 | Collision: No
```

### Metrics Plots
Generated every checkpoint save:
- Episode rewards (with moving average)
- Episode lengths (survival time)
- Training loss over time
- Epsilon decay curve

### ROS Topics to Monitor
```bash
# Camera feed
rostopic hz /rrbot/camera1/image_raw

# Velocity commands
rostopic echo /cmd_vel

# Collision detection
rostopic echo /isHit
```

---

## 🎓 Key Features

✅ **Complete RL Pipeline**: Environment, model, training, inference  
✅ **Robust Reward Design**: Balances multiple objectives  
✅ **Collision Awareness**: Heavy penalty for hitting obstacles  
✅ **Experience Replay**: Efficient learning from past experiences  
✅ **Target Network**: Stable Q-value estimation  
✅ **Checkpointing**: Save/resume training anytime  
✅ **Visualization Tools**: Debug camera processing  
✅ **Comprehensive Logging**: Track all training metrics  
✅ **Easy Interface**: Quickstart script for all operations  

---

## 🔬 Technical Highlights

1. **State Binning**: Reduces 800×800×3 image to 20 features (99.996% reduction)
2. **Reward Shaping**: Carefully tuned multi-objective reward function
3. **Collision Penalty**: -100 reward ensures safety-first learning
4. **Future Discounting**: γ=0.99 balances immediate vs long-term rewards
5. **Exploration Strategy**: Epsilon decay from 1.0 to 0.05 over training
6. **Stability Features**: Target network, gradient clipping, Huber loss

---

## 📚 References & Theory

- **DQN Algorithm**: Mnih et al. (2015) - Human-level control through deep RL
- **Experience Replay**: Lin (1992) - Self-improving reactive agents
- **Target Network**: Reduces moving target problem in Q-learning
- **Epsilon-Greedy**: Balances exploration vs exploitation
- **Reward Shaping**: Guides learning toward desired behavior

---

## 🎯 Performance Goals

After 500 episodes of training, the agent should achieve:
- **Success Rate**: 80-90% course completion
- **Collision Rate**: <10%
- **Average Speed**: 0.4-0.5 m/s
- **Line Following**: Keeps line in center bins 70%+ of time

---

## 🚦 Next Steps for Improvement

1. **Curriculum Learning**: Start simple, gradually increase difficulty
2. **Prioritized Replay**: Sample important experiences more often
3. **Dueling DQN**: Separate value and advantage estimation
4. **Double DQN**: Reduce overestimation bias
5. **Multi-Step Returns**: Better credit assignment with n-step TD
6. **Image Augmentation**: Add robustness to lighting/noise variations

---

## 📝 License & Credits

Part of the ENPH-353 competition environment. Uses ROS Noetic, Gazebo, and PyTorch.

**Created**: November 7, 2025  
**System**: Deep Q-Network for autonomous line following with collision avoidance

---

**Ready to train! Start with: `./quickstart.py train --episodes 500 --render`**
