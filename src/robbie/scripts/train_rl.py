#!/usr/bin/env python3

"""
Main training script for line-following reinforcement learning agent
Trains a DQN agent to navigate the course using camera input
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from rl_environment import LineFollowingEnv
from dqn_model import DQNAgent


class Trainer:
    """
    Training manager for RL line following
    """
    
    def __init__(self, 
                 num_episodes=1000,
                 max_steps=1000,
                 save_freq=50,
                 render=False,
                 checkpoint_dir='checkpoints'):
        """
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            save_freq: Save checkpoint every N episodes
            render: Whether to render camera view during training
            checkpoint_dir: Directory to save checkpoints
        """
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.save_freq = save_freq
        self.render = render
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize environment
        print("Initializing environment...")
        self.env = LineFollowingEnv()
        self.env.max_steps = max_steps
        
        # Initialize agent
        print("Initializing agent...")
        state_size = self.env.get_state_size()
        action_size = self.env.get_action_size()
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.0005,
            gamma=0.99,  # Future reward discount
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            buffer_size=20000,
            batch_size=64,
            target_update_freq=10
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilon_history = []
        
        print(f"Training setup complete!")
        print(f"State size: {state_size}")
        print(f"Action size: {action_size}")
        print(f"Device: {self.agent.device}")
    
    def train(self, resume_from=None):
        """
        Main training loop
        
        Args:
            resume_from: Path to checkpoint to resume from (optional)
        """
        start_episode = 0
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming training from {resume_from}")
            self.agent.load_checkpoint(resume_from)
            start_episode = self.agent.episode_count
            self.load_metrics()
        
        print(f"\nStarting training from episode {start_episode}...")
        print(f"Total episodes: {self.num_episodes}")
        print("=" * 60)
        
        try:
            for episode in range(start_episode, self.num_episodes):
                episode_reward = 0
                episode_loss = []
                
                # Reset environment
                state = self.env.reset(random_start=True)
                
                for step in range(self.max_steps):
                    # Select and execute action
                    action = self.agent.select_action(state, training=True)
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Store experience
                    self.agent.store_experience(state, action, reward, next_state, done)
                    
                    # Train agent
                    loss = self.agent.train_step()
                    if loss is not None:
                        episode_loss.append(loss)
                    
                    episode_reward += reward
                    state = next_state
                    
                    # Render if enabled
                    if self.render:
                        self.env.render()
                    
                    if done:
                        break
                
                # Decay exploration rate
                self.agent.decay_epsilon()
                self.agent.episode_count += 1
                
                # Update target network
                if (episode + 1) % self.agent.target_update_freq == 0:
                    self.agent.update_target_network()
                
                # Record metrics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(step + 1)
                if episode_loss:
                    self.losses.append(np.mean(episode_loss))
                else:
                    self.losses.append(0)
                self.epsilon_history.append(self.agent.epsilon)
                
                # Print progress
                termination_reason = info.get('termination_reason', 'max_steps')
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                print(f"Episode {episode+1}/{self.num_episodes} | "
                      f"Steps: {step+1} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg(100): {avg_reward:.2f} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | "
                      f"Loss: {np.mean(episode_loss) if episode_loss else 0:.4f} | "
                      f"End: {termination_reason}")
                
                # Save checkpoint
                if (episode + 1) % self.save_freq == 0:
                    self.save_checkpoint(episode + 1)
                    self.plot_metrics()
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            self.save_checkpoint('interrupted')
            self.plot_metrics()
        
        finally:
            print("\nCleaning up...")
            self.env.close()
            
            # Final save
            self.save_checkpoint('final')
            self.plot_metrics()
            
            print("Training complete!")
    
    def save_checkpoint(self, episode_id):
        """Save model checkpoint and metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.checkpoint_dir, f'dqn_ep_{episode_id}_{timestamp}.pth')
        self.agent.save_checkpoint(model_path)
        
        # Save metrics
        metrics_path = os.path.join(self.checkpoint_dir, f'metrics_{episode_id}_{timestamp}.json')
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'epsilon_history': self.epsilon_history,
            'agent_metrics': self.agent.get_metrics()
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Checkpoint saved: {model_path}")
    
    def load_metrics(self):
        """Load metrics from most recent checkpoint"""
        # Find most recent metrics file
        metrics_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('metrics_')]
        if not metrics_files:
            return
        
        metrics_files.sort()
        latest_metrics = os.path.join(self.checkpoint_dir, metrics_files[-1])
        
        with open(latest_metrics, 'r') as f:
            metrics = json.load(f)
        
        self.episode_rewards = metrics.get('episode_rewards', [])
        self.episode_lengths = metrics.get('episode_lengths', [])
        self.losses = metrics.get('losses', [])
        self.epsilon_history = metrics.get('epsilon_history', [])
        
        print(f"Loaded metrics from {latest_metrics}")
    
    def plot_metrics(self):
        """Plot training metrics"""
        if not self.episode_rewards:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        if len(self.episode_rewards) >= 10:
            smoothed = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(self.episode_rewards)), smoothed, label='Moving Avg (10)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.3, label='Episode Length')
        if len(self.episode_lengths) >= 10:
            smoothed = np.convolve(self.episode_lengths, np.ones(10)/10, mode='valid')
            axes[0, 1].plot(range(9, len(self.episode_lengths)), smoothed, label='Moving Avg (10)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Training loss
        axes[1, 0].plot(self.losses, alpha=0.6)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].grid(True)
        
        # Epsilon decay
        axes[1, 1].plot(self.epsilon_history)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.checkpoint_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training metrics plot saved to {plot_path}")
        
        # Also try to show (may not work in headless environments)
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            pass


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent for line following')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--save-freq', type=int, default=25, help='Save checkpoint every N episodes')
    parser.add_argument('--render', action='store_true', help='Render camera view during training')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory for checkpoints')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        save_freq=args.save_freq,
        render=args.render,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Start training
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
