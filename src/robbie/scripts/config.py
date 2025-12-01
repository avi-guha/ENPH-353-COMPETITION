"""
Configuration file for RL line following hyperparameters
Edit these values to tune training behavior
"""

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================

ENV_CONFIG = {
    # Camera processing
    'num_bins_per_row': 10,           # Number of horizontal bins (10 = total 20 bins)
    'line_threshold': 180,            # Pixel brightness threshold for line detection
    'bin_activation_threshold': 0.05,  # Min ratio of white pixels to activate bin
    
    # Episode settings
    'max_steps_per_episode': 1000,    # Maximum steps before timeout
    
    # Starting positions (x, y, z, yaw)
    'start_positions': [
        {'x': -5.0, 'y': 0.0, 'z': 0.15, 'yaw': 0.0},
        {'x': -5.0, 'y': -0.5, 'z': 0.15, 'yaw': 0.0},
        {'x': -5.0, 'y': 0.5, 'z': 0.15, 'yaw': 0.0},
    ],
    
    # Robot name in Gazebo (B1 is used in the simulation)
    'robot_name': 'B1',
}


# ==============================================================================
# ACTION SPACE CONFIGURATION
# ==============================================================================

ACTION_CONFIG = {
    # Linear velocities (m/s) - forward speed options
    'linear_velocities': [0.2, 0.4, 0.6],  # [slow, medium, fast]
    
    # Angular velocities (rad/s) - turning options
    'angular_velocities': [1.5, 0.8, 0.0, -0.8, -1.5],  # [hard_left, left, straight, right, hard_right]
    
    # Total actions = len(linear) × len(angular) = 3 × 5 = 15
}


# ==============================================================================
# REWARD CONFIGURATION
# ==============================================================================

REWARD_CONFIG = {
    # Positive rewards
    'center_bins_strong': 10.0,       # Line in 2-3 center bins (bottom third)
    'center_bins_moderate': 5.0,      # Line in 1 center bin (bottom third)
    'lookahead_reward': 3.0,          # Good lookahead (middle third centered)
    'forward_motion_scale': 2.0,      # Multiplier for forward velocity
    
    # Negative penalties
    'deviation_penalty': -3.0,        # Drifting left/right from center
    'line_lost_penalty': -15.0,       # No line detected at all
    'turning_penalty_scale': -0.5,    # Multiplier for abs(angular_velocity)
    
    # Critical penalty
    'collision_penalty': -100.0,      # Hit obstacle (ends episode)
    
    # Center bin indices for bottom third (0-9)
    'center_bin_start': 4,
    'center_bin_end': 7,              # Bins 4, 5, 6 are "center"
}


# ==============================================================================
# NEURAL NETWORK CONFIGURATION
# ==============================================================================

NETWORK_CONFIG = {
    # Architecture
    'hidden_sizes': [128, 128, 64],   # Hidden layer sizes
    'dropout_rate': 0.2,              # Dropout probability for regularization
    
    # Input/output determined by environment
    # state_size = 20 (10 middle bins + 10 bottom bins)
    # action_size = len(linear_vels) × len(angular_vels)
}


# ==============================================================================
# DQN AGENT CONFIGURATION
# ==============================================================================

AGENT_CONFIG = {
    # Learning parameters
    'learning_rate': 0.0005,          # Adam optimizer learning rate
    'gamma': 0.99,                    # Discount factor for future rewards
    
    # Exploration parameters
    'epsilon_start': 1.0,             # Initial exploration rate
    'epsilon_end': 0.05,              # Minimum exploration rate
    'epsilon_decay': 0.995,           # Decay rate per episode
    
    # Experience replay
    'buffer_size': 20000,             # Maximum experiences to store
    'batch_size': 64,                 # Batch size for training
    
    # Training stability
    'target_update_freq': 10,         # Update target network every N episodes
    'gradient_clip_value': 10.0,      # Gradient clipping threshold
    
    # Device
    'device': 'auto',                 # 'auto', 'cuda', or 'cpu'
}


# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

TRAINING_CONFIG = {
    # Training parameters
    'num_episodes': 500,              # Total training episodes
    'save_frequency': 25,             # Save checkpoint every N episodes
    'render_during_training': False,  # Show visualization (slows training)
    
    # Random starting positions
    'random_start': True,             # Randomize start position each episode
    
    # Checkpoint settings
    'checkpoint_dir': 'checkpoints',  # Directory for saving models
    'save_metrics': True,             # Save JSON metrics
    'save_plots': True,               # Generate training plots
}


# ==============================================================================
# INFERENCE CONFIGURATION
# ==============================================================================

INFERENCE_CONFIG = {
    # Test parameters
    'num_test_episodes': 5,           # Episodes for evaluation
    'render_during_test': True,       # Show visualization during testing
    'reset_on_collision': True,       # Reset episode on collision
    
    # Use greedy policy (no exploration) during inference
    'epsilon': 0.0,
}


# ==============================================================================
# VISUALIZATION CONFIGURATION
# ==============================================================================

VISUALIZATION_CONFIG = {
    # Display settings
    'show_bins': True,                # Draw bin boundaries on image
    'show_state_text': True,          # Show state vector as text
    'show_statistics': True,          # Show line position stats
    
    # Colors (BGR format for OpenCV)
    'middle_bin_color': (255, 0, 0),  # Blue for middle third
    'bottom_bin_color': (0, 0, 255),  # Red for bottom third
    'boundary_color': (0, 255, 0),    # Green for boundaries
}


# ==============================================================================
# ADVANCED TUNING GUIDE
# ==============================================================================

"""
QUICK TUNING TIPS:

1. Robot crashes too much:
   - Decrease linear_velocities: [0.15, 0.3, 0.45]
   - Increase collision_penalty: -150.0
   - Increase deviation_penalty: -5.0

2. Too slow/cautious:
   - Increase linear_velocities: [0.3, 0.5, 0.7]
   - Increase forward_motion_scale: 3.0
   - Decrease collision_penalty: -75.0

3. Won't explore enough:
   - Increase epsilon_decay: 0.997 (slower decay)
   - Increase epsilon_end: 0.1
   - Increase buffer_size: 50000

4. Training unstable:
   - Decrease learning_rate: 0.0001
   - Increase batch_size: 128
   - Decrease hidden_sizes: [64, 64, 32]

5. Not learning at all:
   - Check reward function is providing signal
   - Increase learning_rate: 0.001
   - Decrease gamma: 0.95 (focus on immediate rewards)

6. Overfitting:
   - Increase dropout_rate: 0.3
   - Increase buffer_size: 50000
   - Use data augmentation on images

7. Wants more steering precision:
   - Add more angular_velocities: [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0]
   - May slow training due to larger action space

8. Needs smoother driving:
   - Decrease angular_velocities: [1.0, 0.5, 0.0, -0.5, -1.0]
   - Increase turning_penalty_scale: -1.0
"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_all_config():
    """Get complete configuration dictionary"""
    return {
        'environment': ENV_CONFIG,
        'action_space': ACTION_CONFIG,
        'rewards': REWARD_CONFIG,
        'network': NETWORK_CONFIG,
        'agent': AGENT_CONFIG,
        'training': TRAINING_CONFIG,
        'inference': INFERENCE_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
    }


def print_config():
    """Print current configuration"""
    config = get_all_config()
    print("="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    for section, params in config.items():
        print(f"\n{section.upper().replace('_', ' ')}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    print("="*60)


if __name__ == '__main__':
    print_config()
