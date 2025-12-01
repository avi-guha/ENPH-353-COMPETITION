#!/usr/bin/env python3

"""
Inference script for trained line-following agent
Loads a trained DQN model and controls the robot in real-time
"""

import os
import sys
import numpy as np
import rospy
import argparse

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from rl_environment import LineFollowingEnv
from dqn_model import DQNAgent


class InferenceController:
    """
    Real-time controller using trained DQN model
    """
    
    def __init__(self, checkpoint_path, render=True):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            render: Whether to display camera view
        """
        self.checkpoint_path = checkpoint_path
        self.render_enabled = render
        
        # Initialize environment
        print("Initializing environment...")
        self.env = LineFollowingEnv()
        
        # Initialize agent
        print("Initializing agent...")
        state_size = self.env.get_state_size()
        action_size = self.env.get_action_size()
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            epsilon_start=0.0,  # No exploration during inference
            epsilon_end=0.0
        )
        
        # Load trained model
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        self.agent.load_checkpoint(checkpoint_path)
        
        print("Inference controller ready!")
        print(f"State size: {state_size}")
        print(f"Action size: {action_size}")
        print(f"Device: {self.agent.device}")
    
    def run(self, num_episodes=5, reset_on_collision=True):
        """
        Run inference for multiple episodes
        
        Args:
            num_episodes: Number of episodes to run
            reset_on_collision: Whether to reset on collision
        """
        print(f"\nStarting inference for {num_episodes} episodes...")
        print("=" * 60)
        
        episode_rewards = []
        episode_lengths = []
        collisions = 0
        
        try:
            for episode in range(num_episodes):
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                
                # Reset environment
                state = self.env.reset(random_start=(episode > 0))
                episode_reward = 0
                step = 0
                
                # Run episode
                while True:
                    # Select action (greedy, no exploration)
                    action = self.agent.select_action(state, training=False)
                    
                    # Execute action
                    next_state, reward, done, info = self.env.step(action)
                    
                    episode_reward += reward
                    step += 1
                    state = next_state
                    
                    # Render
                    if self.render_enabled:
                        self.env.render()
                    
                    # Print step info
                    if step % 50 == 0:
                        print(f"  Step {step}: Reward={episode_reward:.2f}")
                    
                    # Check termination
                    if done:
                        if info.get('collision', False):
                            collisions += 1
                            print(f"  COLLISION DETECTED!")
                            if reset_on_collision:
                                break
                        elif info.get('timeout', False):
                            print(f"  Episode timeout")
                            break
                        else:
                            print(f"  Episode completed")
                            break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(step)
                
                print(f"Episode {episode + 1} finished:")
                print(f"  Total steps: {step}")
                print(f"  Total reward: {episode_reward:.2f}")
                print(f"  Average reward per step: {episode_reward/step:.2f}")
        
        except KeyboardInterrupt:
            print("\n\nInference interrupted by user!")
        
        finally:
            # Print summary
            print("\n" + "=" * 60)
            print("INFERENCE SUMMARY")
            print("=" * 60)
            print(f"Episodes completed: {len(episode_rewards)}")
            print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
            print(f"Total collisions: {collisions}")
            print(f"Collision rate: {collisions/len(episode_rewards)*100:.1f}%")
            print("=" * 60)
            
            # Cleanup
            self.env.close()
    
    def run_continuous(self):
        """
        Run continuously without resets (for demonstration)
        """
        print("\nStarting continuous inference...")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        state = self.env.reset()
        total_reward = 0
        step = 0
        
        try:
            while not rospy.is_shutdown():
                # Select and execute action
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                total_reward += reward
                step += 1
                state = next_state
                
                # Render
                if self.render_enabled:
                    self.env.render()
                
                # Print periodic updates
                if step % 100 == 0:
                    print(f"Step {step}: Total reward={total_reward:.2f}, Avg={total_reward/step:.2f}")
                
                # Handle collisions
                if done and info.get('collision', False):
                    print(f"\nCollision at step {step}! Resetting...")
                    state = self.env.reset()
                    
        except KeyboardInterrupt:
            print("\n\nStopped by user!")
        
        finally:
            print(f"\nFinal stats:")
            print(f"  Total steps: {step}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Average reward: {total_reward/step:.2f}")
            self.env.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run trained DQN agent for line following')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--continuous', action='store_true', help='Run continuously without episode resets')
    parser.add_argument('--no-render', action='store_true', help='Disable camera view rendering')
    parser.add_argument('--no-reset-on-collision', action='store_true', help='Continue after collision')
    
    args = parser.parse_args()
    
    # Create controller
    controller = InferenceController(
        checkpoint_path=args.checkpoint,
        render=not args.no_render
    )
    
    # Run
    if args.continuous:
        controller.run_continuous()
    else:
        controller.run(
            num_episodes=args.episodes,
            reset_on_collision=not args.no_reset_on_collision
        )


if __name__ == '__main__':
    main()
