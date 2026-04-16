import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import argparse
import json
import yaml

def set_seed(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        """Update target network gradually using Polyak averaging."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    if len(ret) >= n:
        ret[n:] = ret[n:] - ret[:-n]
        res = ret / n
        res[:n-1] = ret[:n-1] / np.arange(1, n)
        return res
    return ret / np.arange(1, len(ret) + 1)

def evaluate_agent(env, agent, eval_episodes=30):
    """
    Executes a pure test phase: the agent plays without exploration (Greedy)
    and without updating networks or saving anything in the Replay Buffer.
    """
    eval_rewards = []
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done, truncated = False, False
        tot_rew = 0
        while not (done or truncated):
            # eval_mode = True bypasses Softmax (lethal actions impossible if max Q-value differs)
            action = agent.act(state, eval_mode=True)
            state, reward, done, truncated, _ = env.step(action)
            tot_rew += reward
        eval_rewards.append(tot_rew)
    
    return np.mean(eval_rewards), np.std(eval_rewards)

def train_agent(env, agent, episodes, log_dir, timestamp, window_width=100, eval_freq=50, eval_episodes=30):
    """
    Executes the core training loop and invokes periodic evaluations.
    """
    rewards_list = []
    eval_history = []  # Stores tuples (episode, mean_reward, std_reward)

    for e in range(episodes):
        state, info = env.reset()
        action = agent.act(state) # Chooses first action (useful for pure Sarsa)
        
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Interaction
            next_state, reward, done, truncated, info = env.step(action)
            next_action = agent.act(next_state) if not (done or truncated) else None
            
            # Save into buffer
            agent.memory.append(state, action, reward, next_state, (done or truncated), next_action)
            
            # Explicit learning on a mini-batch (Actual Training)
            agent.learn()

            state = next_state
            action = next_action 
            total_reward += reward

        # Temperature Decay at episode end
        agent.update_temp()
        rewards_list.append(total_reward)

        # Compute Moving Average
        avg_reward = np.mean(rewards_list[-window_width:]) if len(rewards_list) >= window_width else np.mean(rewards_list)
        
        # LOGGING TRAINING
        log_msg = f"Episode: {e+1:03d}/{episodes} | Reward: {total_reward: 7.2f} | Temp: {agent.temp:.3f} | MA ({window_width} ep): {avg_reward: 7.2f}"
        print(log_msg)
        logging.info(log_msg)
        
        # ======= EVALUATION PHASE =======
        # Suspend training every 'eval_freq' episodes to evaluate the 'cold' policy 
        if (e + 1) % eval_freq == 0:
            logging.info(f"--- Starting Evaluation run at Episode {e+1} ---")
            mean_eval, std_eval = evaluate_agent(env, agent, eval_episodes)
            eval_history.append((e+1, mean_eval, std_eval))
            
            eval_log_msg = f"*** EVALUATION | Epi: {e+1:03d} | Mean Reward ({eval_episodes} Run): {mean_eval:7.2f} ± {std_eval:5.2f} ***"
            print(f"\033[93m{eval_log_msg}\033[0m") # Yellow color in terminal
            logging.info(eval_log_msg)
            logging.info("--- End Evaluation ---")

    return rewards_list, eval_history

def set_default_config(cfg, parsed_args):
    """
    Overwrites YAML config parameters with command-line arguments (if present).
    Priority: argparse > YAML config > hardcoded default values.
    """
    for key, value in vars(parsed_args).items():
        if value is not None:
            setattr(cfg, key, value)

def main():
    set_seed(42) # Force determinism for reproducibility

    args = argparse.ArgumentParser(description="Deep RL Agent for LunarLander using PyTorch")
    args.add_argument("--from-config", type=str, default=os.path.join("config", "config.yaml"), help="YAML configuration file")
    args.add_argument("--algorithm", type=str, choices=["q_learning", "sarsa", "expected_sarsa"], help="Overrides algorithm from config")
    args.add_argument("--episodes", type=int, help="Overrides episodes from config")
    args.add_argument("--runs", type=int, help="Overrides runs from config")
    args.add_argument("--output-dir", type=str, help="Output directory")
    parsed_args = args.parse_args()
    
    # Load and prioritize YAML config vs argparse
    with open(parsed_args.from_config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        
    cfg = argparse.Namespace(**cfg_dict)
    set_default_config(cfg, parsed_args)
    
    # Setup logging
    log_dir = cfg.output_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=f"{log_dir}/training_{timestamp}.log",
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    logging.info(f"Starting training with Algorithm: {cfg.algorithm} from YAML")
    
    # Auto-save parameters at run start
    config_dir = f"{log_dir}"
    config_data = vars(cfg)
    
    config_path = f"{config_dir}/config_{cfg.algorithm}_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
        
    logging.info(f"Execution configuration saved in {config_path}")
    print(f"Configuration saved in {config_path}")
    
    all_runs_rewards = []
    all_runs_eval_means = []
    eval_eps = []

    for run in range(1, cfg.runs + 1):
        logging.info(f"========== STARTING RUN {run}/{cfg.runs} ==========")
        print(f"\n========== STARTING RUN {run}/{cfg.runs} ==========")
        
        # Initialize a new environment and new agent over each RUN
        env_run = gym.make("LunarLander-v3")
        state_dim = env_run.observation_space.shape[0]
        action_dim = env_run.action_space.n
        
        agent = LanderAgent(
            state_dim, 
            action_dim, 
            algorithm=cfg.algorithm,
            lr=cfg.lr,
            gamma=cfg.gamma,
            tau=cfg.tau,
            batch_size=cfg.batch_size,
            buffer_capacity=cfg.buffer_capacity,
            hidden_size=cfg.hidden_size,
            temp_init=cfg.temp_init,
            temp_min=cfg.temp_min,
            temp_decay=cfg.temp_decay,
        )
        
        rewards_list, eval_history = train_agent(
            env=env_run,
            agent=agent,
            episodes=cfg.episodes,
            log_dir=log_dir,
            timestamp=timestamp,
            window_width=cfg.window_width,
            eval_freq=cfg.eval_freq,
            eval_episodes=cfg.eval_episodes
        )

        env_run.close()
        
        # Save model for current run (needed for Ensemble or Best Seed extraction)
        run_model_path = f"{log_dir}/model_{cfg.algorithm}_run{run}_{timestamp}.pth"
        torch.save(agent.q_network.state_dict(), run_model_path)
        logging.info(f"Model run {run} saved in {run_model_path}.")
        
        all_runs_rewards.append(rewards_list)
        if eval_history:
            if run == 1:
                eval_eps = [x[0] for x in eval_history]
            all_runs_eval_means.append([x[1] for x in eval_history])

    # == AGGREGATION AND SAVING FINAL PLOTS ==
    plt.figure(figsize=(10,6))
    
    all_runs_rewards = np.array(all_runs_rewards)
    mean_train_rewards = np.mean(all_runs_rewards, axis=0)
    std_train_rewards = np.std(all_runs_rewards, axis=0)
    
    plt.plot(mean_train_rewards, label='Mean Training Reward', alpha=0.3, color='blue')
    if cfg.runs > 1:
        plt.fill_between(range(cfg.episodes), mean_train_rewards - std_train_rewards, mean_train_rewards + std_train_rewards, alpha=0.1, color='blue')
    
    ma_train = moving_average(mean_train_rewards, n=cfg.window_width)
    plt.plot(ma_train, label=f"Mean Training MA ({cfg.window_width} Ep)", color='red')
    
    if len(all_runs_eval_means) > 0:
        all_runs_eval_means = np.array(all_runs_eval_means)
        global_eval_means = np.mean(all_runs_eval_means, axis=0)
        global_eval_stds = np.std(all_runs_eval_means, axis=0)
        
        plt.errorbar(eval_eps, global_eval_means, yerr=global_eval_stds, fmt='-o', color='gold', ecolor='darkorange',
                     linewidth=2, capsize=4, label=f'Mean Eval Score ({cfg.eval_episodes} Ep, Greedy)')
    
    plt.axhline(200, color='green', linestyle='--', label='Solution threshold')
    plt.xlabel('Episodes')
    plt.ylabel(f'Total reward (averaged over {cfg.runs} runs)')
    plt.title(f'Learning curve - {cfg.algorithm} (mean of {cfg.runs} runs)' if cfg.runs > 1 else f'Learning curve with periodic evaluation - {cfg.algorithm}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plot_path = f"{log_dir}/reward_plot_{cfg.algorithm}_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Plot saved in: {plot_path}")
    logging.info(f"Plot saved in: {plot_path}")

    print(f"\nTraining completed for {cfg.algorithm} with {cfg.runs} runs. Final plot saved in {plot_path}.")
    logging.info(f"Training completed for {cfg.algorithm} with {cfg.runs} runs. Final plot saved in {plot_path}.")
    
if __name__ == "__main__":
    main()
