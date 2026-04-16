import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    """
    Small MLP to approximate the Action-Value function Q(s, a).
    """
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience Replay Buffer to break transition correlations."""
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def append(self, state, action, reward, next_state, finished, next_action=None):
        self.buffer.append((state, action, reward, next_state, finished, next_action))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, finished, next_actions = zip(*batch)
        
        # Filter None next_actions (e.g. at terminal states)
        next_acts = [a if a is not None else 0 for a in next_actions]
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(finished, dtype=np.float32),
            np.array(next_acts)
        )

    def __len__(self):
        return len(self.buffer)

class LanderAgent:
    """
    Deep RL Agent that supports Q-Learning, SARSA, and Expected SARSA.
    """
    def __init__(self, state_dim, action_dim, algorithm, lr, gamma, tau, batch_size, buffer_capacity, hidden_size, temp_init, temp_min, temp_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.temp = temp_init           # Initial exploration temperature
        self.temp_min = temp_min        # Minimum temperature for exploration (prevents collapse to greedy policy)
        self.temp_decay = temp_decay    # Temperature decay PER EPISODE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Main and Target networks (improves stability)
        self.q_network = QNetwork(state_dim, action_dim, hidden_size=hidden_size).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_size=hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # PyTorch Optimizer (Gradient Clipping added later for stability)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.memory = ReplayBuffer(capacity=buffer_capacity, batch_size=batch_size)
        
    def act(self, state, eval_mode=False):
        """Softmax policy for training or pure greedy policy for Evaluation."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval() # Temporarily disable gradients/dropout
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)
        self.q_network.train()
        
        if eval_mode:
            # Deterministic policy without exploration
            return torch.argmax(q_values).item()
        
        # Softmax with temperature (and numerical stability trick)
        preferences = q_values / self.temp
        max_pref = torch.max(preferences)
        exp_pref = torch.exp(preferences - max_pref)
        action_probs = (exp_pref / torch.sum(exp_pref)).cpu().numpy()
        
        return np.random.choice(self.action_dim, p=action_probs)

    def learn(self):
        """Explicit learning step for Q-Learning, SARSA, and Expected SARSA."""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, finished, next_actions = self.memory.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        finished = torch.FloatTensor(finished).unsqueeze(1).to(self.device)
        next_actions = torch.LongTensor(next_actions).unsqueeze(1).to(self.device)

        # 1. Compute current Q-Values
        q_values = self.q_network(states).gather(1, actions)

        # 2. Compute target Q-Values based on the algorithm
        with torch.no_grad():
            if self.algorithm == "q_learning":
                # Q-Learning: Use max Q-value for next state
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                
            elif self.algorithm == "sarsa":
                # SARSA: Use 'next_action' from buffer to calculate target
                next_q_values = self.target_network(next_states).gather(1, next_actions)
                
            elif self.algorithm == "expected_sarsa":
                # Expected SARSA: Weighted average by Softmax probabilities
                q_next_full = self.target_network(next_states)
                
                # Compute probabilities using batched Softmax policy
                preferences = q_next_full / self.temp
                max_prefs = torch.max(preferences, dim=1, keepdim=True)[0]
                exp_prefs = torch.exp(preferences - max_prefs)
                probs = exp_prefs / torch.sum(exp_prefs, dim=1, keepdim=True)
                
                next_q_values = (probs * q_next_full).sum(dim=1, keepdim=True)
                
            else:
                raise ValueError("Unknown algorithm. Choose among Q-Learning, SARSA, Expected Sarsa.")

            target_q_values = rewards + (self.gamma * next_q_values * (1 - finished))

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # Target network soft update
        self.soft_update(self.q_network, self.target_network, self.tau)

    def update_temp(self):
        # Temperature decay called at episode end
        if self.temp > self.temp_min:
            self.temp *= self.temp_decay

    def soft_update(self, local_model, target_model, tau):
        """Update target network gradually using Polyak averaging (EMA)"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
