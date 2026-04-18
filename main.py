import gymnasium as gym
import torch
import numpy as np
import os
import logging
from datetime import datetime
import argparse
import json
import yaml
import warnings

# Suppress pkg_resources deprecation warning originating from pygame/gymnasium
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
from src.agent import LanderAgent
from src.utils import plot_training_curve, set_seed, set_default_config, setup_logger
from src.train import train_agent

def single_run(run, cfg, timestamp):
    logging.info(f"\n========== STARTING RUN {run}/{cfg.runs} ==========")
    
    # Environment initialization
    env_run = gym.make("LunarLander-v3")
    state_dim = env_run.observation_space.shape[0]
    action_dim = env_run.action_space.n
    
    # Agent initialization
    agent = LanderAgent(
        state_dim, 
        action_dim, 
        algorithm=cfg.algorithm,
        double_learning=cfg.double,
        lr=cfg.lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        batch_size=cfg.batch_size,
        buffer_capacity=cfg.buffer_capacity,
        hidden_size=cfg.hidden_size,
        loss=cfg.loss,
        temp_init=cfg.temp_init,
        temp_min=cfg.temp_min,
        temp_decay=cfg.temp_decay,
    )
    
    # Training loop with periodic evaluation
    rewards_list, eval_history = train_agent(
        env=env_run,
        agent=agent,
        episodes=cfg.episodes,
        window_width=cfg.window_width,
        eval_period=cfg.eval_period,
        eval_runs=cfg.eval_runs,
        run_index=run,
        train_runs=cfg.runs
    )

    env_run.close()
    
    # Save model for current run (needed for Ensemble or Best Seed extraction)
    run_model_path = os.path.join(cfg.output_dir, f"model_{cfg.algorithm}_run{run}_{timestamp}.pth")
    torch.save(agent.q_network.state_dict(), run_model_path)
    logging.info(f"Model run {run} saved in {run_model_path}.")
        
    return rewards_list, eval_history

def main():
    # Reproducibility
    set_seed(42)

    # Arguments from CLI and YAML config loading
    args = argparse.ArgumentParser(description="Deep RL Agent for LunarLander using PyTorch")
    args.add_argument("--from-config", type=str, default=os.path.join("config", "config.yaml"), help="YAML configuration file")
    args.add_argument("--algorithm", type=str, choices=["q_learning", "sarsa", "expected_sarsa"], help="Overrides algorithm from config")
    args.add_argument("--double", action='store_true', help="Use Double Q-Learning (or Double SARSA/Expected SARSA) if set")
    args.add_argument("--episodes", type=int, help="Number of training episodes")
    args.add_argument("--lr", type=float, help="Learning rate")
    args.add_argument("--temp-decay", type=float, help="Temperature decay rate")
    args.add_argument("--temp-min", type=float, help="Minimum temperature for exploration")
    args.add_argument("--eval-period", type=int, help="Frequency of evaluation (in episodes)")
    args.add_argument("--eval-runs", type=int, help="Number of evaluation episodes per evaluation phase")
    args.add_argument("--runs", type=int, help="Number of independent runs for averaging results")
    args.add_argument("--hidden-size", type=int, help="Number of hidden units in the Q-network")
    args.add_argument("--batch-size", type=int, help="Batch size for training")
    args.add_argument("--buffer-capacity", type=int, help="Capacity of the replay buffer")
    args.add_argument("--loss", type=str, choices=["mse", "smooth_l1"], help="Loss function for training")
    args.add_argument("--output-dir", type=str, help="Output directory")
    parsed_args = args.parse_args()
    
    with open(parsed_args.from_config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        
    cfg = argparse.Namespace(**cfg_dict)
    set_default_config(cfg, parsed_args)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(cfg, timestamp)
    
    logging.info(f"Starting training with algorithm: '{cfg.algorithm}'")
    
    # Auto-save parameters at run start
    config_data = vars(cfg)
    config_path = os.path.join(cfg.output_dir, f"config_{cfg.algorithm}_{timestamp}.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
        
    logging.info(f"Execution configuration saved in {config_path}")
    
    # Main loop over multiple runs
    rewards, eval_means, eval_eps = [], [], []
    for run in range(1, cfg.runs + 1):
        rewards_list, eval_history = single_run(run, cfg, timestamp)
        
        rewards.append(rewards_list)
        eval_means.append([x[1] for x in eval_history])
        eval_eps.append([x[0] for x in eval_history])

    # Aggregation and final plotting
    rewards = np.array(rewards)
    mean_train_rewards = np.mean(rewards, axis=0)
    std_train_rewards = np.std(rewards, axis=0)
    
    plot_training_curve(mean_train_rewards, std_train_rewards, eval_means, eval_eps, cfg, timestamp)
    
    logging.info(f"\nTraining completed for {cfg.algorithm} with {cfg.runs} runs")
    
if __name__ == "__main__":
    main()
