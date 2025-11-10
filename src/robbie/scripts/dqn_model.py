#!/usr/bin/env python3

"""
Deep Q-Network (DQN) implementation for line following
Uses a neural network to approximate Q-values for state-action pairs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(nn.Module):
    """
    Deep Q-Network architecture
    Takes binned camera state (20 features) as input
    Outputs Q-values for each action
    """
    
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128, 64]):
        """
        Args:
            state_size: Size of input state (20 bins)
            action_size: Number of possible actions (15 for 3 speeds x 5 steerings)
            hidden_sizes: List of hidden layer sizes
        """
        super(DQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Prevent overfitting
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """
        Forward pass through network
        
        Args:
            state: Tensor of shape (batch_size, state_size) or (state_size,)
            
        Returns:
            q_values: Tensor of shape (batch_size, action_size) or (action_size,)
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """
    
    def __init__(self, capacity=10000):
        """
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample random batch of experiences
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent that learns to follow lines using reinforcement learning
    """
    
    def __init__(self, state_size, action_size, 
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 batch_size=64,
                 target_update_freq=10,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            state_size: Size of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency (in episodes) to update target network
            device: 'cuda' or 'cpu'
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Q-Network and Target Network
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is only used for inference
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.training_step = 0
        self.episode_count = 0
        
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state (numpy array)
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            action: Integer action index
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: choose best action according to Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one step of training on a batch from replay buffer
        
        Returns:
            loss: Training loss (None if not enough experiences)
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss (Huber loss is more robust than MSE)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_checkpoint(self, filepath):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        print(f"Checkpoint loaded from {filepath}")
    
    def get_metrics(self):
        """Return current training metrics"""
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'buffer_size': len(self.memory)
        }


if __name__ == '__main__':
    # Test the DQN architecture
    state_size = 20
    action_size = 15
    
    print("Testing DQN architecture...")
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    print(f"Q-Network architecture:")
    print(agent.q_network)
    print(f"\nDevice: {agent.device}")
    print(f"Number of parameters: {sum(p.numel() for p in agent.q_network.parameters())}")
    
    # Test forward pass
    dummy_state = np.random.rand(state_size)
    action = agent.select_action(dummy_state)
    print(f"\nTest state: {dummy_state}")
    print(f"Selected action: {action}")
    
    # Test experience storage and training
    for i in range(100):
        state = np.random.rand(state_size)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.rand(state_size)
        done = random.random() < 0.1
        agent.store_experience(state, action, reward, next_state, done)
    
    print(f"\nBuffer size: {len(agent.memory)}")
    
    # Test training step
    loss = agent.train_step()
    print(f"Training loss: {loss}")
    
    print("\nDQN architecture test completed successfully!")
