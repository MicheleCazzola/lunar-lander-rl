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
    def __init__(self, state_dim, action_dim, algorithm, double_learning, lr, gamma, tau, batch_size, buffer_capacity, hidden_size, loss, temp_init, temp_min, temp_decay):
        
        assert algorithm in ["q_learning", "sarsa", "expected_sarsa"], "Invalid algorithm choice"
        assert loss in ["mse", "smooth_l1"], "Invalid loss function choice"
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.double_learning = double_learning
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.temp = temp_init               # Initial exploration temperature
        self.temp_min = temp_min            # Minimum temperature for exploration (prevents collapse to greedy policy)
        self.temp_decay = temp_decay        # Temperature decay per episode
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Main and Target networks (improves stability)
        self.q_network = QNetwork(state_dim, action_dim, hidden_size=hidden_size).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_size=hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # PyTorch Optimizer (Gradient Clipping added later for stability)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.memory = ReplayBuffer(capacity=buffer_capacity, batch_size=batch_size)
        self.loss_fn = nn.MSELoss() if loss == "mse" else nn.SmoothL1Loss()
        
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
        if len(self.memory) < max(self.batch_size, 500):
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
                if self.double_learning:
                    # Double Q-Learning: Action selection from local, evaluation from target
                    next_q_local = self.q_network(next_states)
                    next_actions_local = torch.argmax(next_q_local, dim=1, keepdim=True)
                    next_q_values = self.target_network(next_states).gather(1, next_actions_local)
                else:
                    next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            
            elif self.algorithm == "sarsa":
                # SARSA: Use 'next_action' from buffer to calculate target
                if self.double_learning:
                    # Double SARSA: Action selection from local, evaluation from target
                    next_q_values = self.target_network(next_states).gather(1, next_actions)
                else:
                    next_q_values = self.q_network(next_states).gather(1, next_actions)
                    
            elif self.algorithm == "expected_sarsa":
                # Double Expected SARSA: Weighted average by Softmax probabilities
                if self.double_learning:
                    q_next_target = self.target_network(next_states)
                    q_next_local = self.q_network(next_states)
                else:
                    q_next_target = self.target_network(next_states)
                    q_next_local = q_next_target
                
                # Compute probabilities using batched Softmax policy based on local network
                preferences = q_next_local / self.temp
                max_prefs = torch.max(preferences, dim=1, keepdim=True)[0]
                exp_prefs = torch.exp(preferences - max_prefs)
                probs = exp_prefs / torch.sum(exp_prefs, dim=1, keepdim=True)
                
                next_q_values = (probs * q_next_target).sum(dim=1, keepdim=True)
                
            else:
                raise ValueError("Unknown algorithm. Choose among Q-Learning, SARSA, Expected Sarsa.")

            target_q_values = rewards + (self.gamma * next_q_values * (1 - finished))

        loss = self.loss_fn(q_values, target_q_values)
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


class ActorNetwork(nn.Module):
    """
    MLP to approximate the Policy pi(a|s).
    It outputs numerical preferences (logits) for the action probabilities.
    """
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Raw logits


class CriticNetwork(nn.Module):
    """
    MLP to approximate the State-Value function V(s).
    It outputs a single numerical value estimating the expected return.
    """
    def __init__(self, state_dim, hidden_size=64):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RolloutTrajectoryBuffer:
    """
    A temporary buffer to store the trajectory.
    Exposes an identical method signature to ReplayBuffer to be compatible with main.py logic.
    For on-policy algorithms, it clears after each use.
    """
    def __init__(self):
        self.buffer = []

    def append(self, state, action, reward, next_state, finished, next_action=None):
        self.buffer.append((state, action, reward, next_state, finished))

    def sample(self):
        states, actions, rewards, next_states, finished = zip(*self.buffer)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(finished, dtype=np.float32)
        )

    def clear(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)


class ActorCriticAgent:
    """
    On-Policy Softmax Actor-Critic Agent.
    Updates the Actor using Policy Gradient guided by Advantage (TD Error),
    and updates the Critic using TD Error.
    """
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, tau, hidden_size, loss, temp_init, temp_min, temp_decay, entropy_coef=0.0, use_average_reward=False, avg_reward_alpha=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.temp = temp_init
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        self.entropy_coef = entropy_coef
        self.use_average_reward = use_average_reward
        self.avg_reward_alpha = avg_reward_alpha
        self.avg_reward = 0.0
        self.loss_fn = nn.MSELoss() if loss == "mse" else nn.SmoothL1Loss()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_size).to(self.device)
        self.target_critic = CriticNetwork(state_dim, hidden_size).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # For Actor-Critic we use our Rollout buffer since it is on-policy
        self.memory = RolloutTrajectoryBuffer()

    def act(self, state, eval_mode=False):
        """Softmax preference policy for acting."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(state_tensor).squeeze(0)
        self.actor.train()
        
        if eval_mode:
            return torch.argmax(logits).item()
            
        # Numerically stable Softmax adjusted by temperature
        preferences = logits / self.temp
        max_pref = torch.max(preferences)
        exp_prefs = torch.exp(preferences - max_pref)
        action_probs = (exp_prefs / torch.sum(exp_prefs)).cpu().numpy()
        
        return np.random.choice(self.action_dim, p=action_probs)

    def learn(self):
        """Updates both Critic and Actor from collected rollout data using TD error."""
        if len(self.memory) == 0:
            return
            
        states, actions, rewards, next_states, finished = self.memory.sample()
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        finished = torch.FloatTensor(finished).unsqueeze(1).to(self.device)

        # ----------------------------------------------------
        # Critic Update
        # ----------------------------------------------------
        # Get V(s)
        v_s = self.critic(states)
        
        # Get target V(s') using the target network
        with torch.no_grad():
            v_next_s = self.target_critic(next_states)
            if self.use_average_reward:
                td_target = (rewards - self.avg_reward) + v_next_s * (1 - finished)
            else:
                td_target = rewards + self.gamma * v_next_s * (1 - finished)
        
        critic_loss = self.loss_fn(v_s, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ----------------------------------------------------
        # Actor Update
        # ----------------------------------------------------
        v_s_detached = v_s.detach()
        td_error = (td_target - v_s_detached).squeeze()
        
        # Update average reward if using average reward formulation
        if self.use_average_reward:
            self.avg_reward += self.avg_reward_alpha * td_error.mean().item()

        logits = self.actor(states)
        preferences = logits / self.temp
        action_probs = F.softmax(preferences, dim=-1)
        
        # log pi(a|s)
        log_probs = torch.log(action_probs.gather(1, actions) + 1e-10).squeeze()
        
        # Entropy regularization
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1).mean()
        
        # Policy gradient: grad [ log pi(a|s) * delta ] -> to maximize we minimize its negative
        actor_loss = -(log_probs * td_error).mean() - self.entropy_coef * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Target network soft update for the Critic
        self.soft_update(self.critic, self.target_critic, self.tau)

        # Clear buffer since this is strictly on-policy
        self.memory.clear()

    def update_temp(self):
        # Temperature decay called at episode end
        if self.temp > self.temp_min:
            self.temp *= self.temp_decay

    def soft_update(self, local_model, target_model, tau):
        """Update target network gradually using Polyak averaging (EMA)"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

