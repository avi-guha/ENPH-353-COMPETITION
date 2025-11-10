# 🤖 Reinforcement Learning Line Following System

## 📖 Documentation Index

**Start here** based on what you need:

### 🚀 I want to get started quickly
→ Read **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** (3 min read)

### 📚 I want the full user guide
→ Read **[RL_README.md](RL_README.md)** (15 min read)

### 🔬 I want technical details
→ Read **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** (20 min read)

### 🎯 I want everything at once
→ Read **[COMPLETE_OVERVIEW.md](COMPLETE_OVERVIEW.md)** (10 min read)

---

## 🎮 Quick Commands

```bash
# Train a model
./scripts/quickstart.py train --episodes 500 --render

# Test trained model
./scripts/quickstart.py test

# Debug camera processing
./scripts/quickstart.py visualize

# Check system status
./scripts/quickstart.py info
```

---

## 📂 File Structure

```
robbie/
├── 📖 Documentation
│   ├── QUICK_REFERENCE.md       ← 1-page cheat sheet
│   ├── RL_README.md            ← Complete user guide  
│   ├── PROJECT_SUMMARY.md      ← Technical deep dive
│   ├── COMPLETE_OVERVIEW.md    ← Full system overview
│   └── INDEX.md                ← This file
│
├── 🐍 Core Scripts
│   ├── rl_environment.py       ← Environment wrapper
│   ├── dqn_model.py            ← Neural network
│   ├── train_rl.py             ← Training script
│   ├── run_inference.py        ← Testing script
│   ├── visualize_bins.py       ← Debug tool
│   ├── quickstart.py           ← CLI interface
│   └── config.py               ← Hyperparameters
│
├── 🔧 Configuration
│   ├── requirements.txt        ← Python dependencies
│   └── launch/
│       └── rl_training.launch  ← ROS launch file
│
├── 🤖 Robot Definition
│   └── urdf/
│       └── robbie.xacro        ← Robot URDF (with collision sensor)
│
└── 💾 Generated (during training)
    └── checkpoints/
        ├── dqn_ep_*.pth        ← Model checkpoints
        ├── metrics_*.json      ← Training data
        └── training_metrics.png ← Visualizations
```

---

## 🎯 What This System Does

Uses **Deep Q-Network (DQN)** reinforcement learning to:
1. ✅ Process camera images into 20 binary features
2. ✅ Learn optimal steering and speed control
3. ✅ Follow white/yellow lines smoothly
4. ✅ Avoid collisions with obstacles (heavy penalty)
5. ✅ Navigate the course autonomously

---

## 🧠 Key Features

- **Intelligent State Representation**: Camera → 20 bins (99.996% compression)
- **Multi-Objective Rewards**: Line following + speed + smoothness + safety
- **Safety First**: -100 penalty for collisions ensures cautious behavior
- **Production Ready**: Checkpointing, logging, visualization, documentation
- **Easy to Use**: Single command to train, single command to test

---

## 📊 Expected Performance

After **500 episodes** (~3-6 hours training):
- ✅ 80%+ success rate (completes course without collision)
- ✅ 0.4-0.5 m/s average speed
- ✅ Smooth steering with good line centering
- ✅ <10% collision rate

---

## 🚦 Training Phases

| Phase | Episodes | Behavior | Avg Reward |
|-------|----------|----------|------------|
| **Exploration** | 1-100 | Random actions, crashes | -50 to 100 |
| **Learning** | 100-300 | Basic steering, some crashes | 100 to 500 |
| **Refinement** | 300-500 | Smooth navigation, rare crashes | 500+ |

---

## 🔧 Quick Customization

Edit `scripts/config.py` to change:
- Robot speeds
- Steering angles  
- Reward values
- Neural network size
- Training parameters
- And more!

---

## 🆘 Need Help?

1. **System status**: Run `./scripts/quickstart.py info`
2. **Camera issues**: Run `./scripts/quickstart.py visualize`
3. **ROS problems**: Check `rostopic list`
4. **Training issues**: See troubleshooting in `RL_README.md`

---

## 📦 What You Get

- ✅ **7 Python scripts** (2,700+ lines of production code)
- ✅ **4 documentation files** (comprehensive guides)
- ✅ **Complete DQN implementation** (experience replay, target network, etc.)
- ✅ **Visualization tools** (debug camera processing)
- ✅ **Configuration system** (easy hyperparameter tuning)
- ✅ **Checkpoint management** (save/resume training)

---

## 🎓 Technologies Used

- **ROS Noetic**: Robot Operating System
- **Gazebo**: 3D robot simulator
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision
- **Reinforcement Learning**: DQN algorithm

---

## 🎬 One-Line Quick Start

```bash
cd ~/ENPH-353-COMPETITION/src/robbie/scripts && ./quickstart.py train --episodes 500
```

That's it! The system handles everything else automatically.

---

## 📈 Monitoring Training

Watch these metrics in the terminal output:
- **Episode Reward**: Should increase over time
- **Collision Rate**: Should decrease
- **Epsilon**: Should decay from 1.0 to 0.05
- **Loss**: Should stabilize (not explode)

Plots are auto-saved to `checkpoints/training_metrics.png`

---

## 🏆 Success Criteria

Your agent is well-trained when:
- ✅ Average reward > 500
- ✅ Episode length > 800 steps
- ✅ Collision rate < 10%
- ✅ Line stays in center bins 70%+ of time

---

## 🔮 Next Steps After Training

1. **Test the model**: `./quickstart.py test`
2. **Tune parameters**: Edit `config.py` and retrain
3. **Visualize behavior**: Use `--render` flag
4. **Deploy**: Run `./quickstart.py test --continuous`

---

## 📞 Documentation Links

- [Quick Reference](QUICK_REFERENCE.md) - Cheat sheet
- [User Guide](RL_README.md) - Full tutorial
- [Technical Summary](PROJECT_SUMMARY.md) - Deep dive
- [Complete Overview](COMPLETE_OVERVIEW.md) - Everything

---

**🚀 Ready to train your autonomous line-following robot? Start with the [Quick Reference](QUICK_REFERENCE.md)!**

---

*Deep Q-Network for Autonomous Line Following with Collision Avoidance*  
*Created: November 7, 2025*
