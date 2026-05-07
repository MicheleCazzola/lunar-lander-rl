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
from src.agent import LanderAgent, ActorCriticAgent
from src.utils import plot_training_curve, save_results, set_seed, set_default_config, setup_logger
from src.train import train_agent

def single_run(run, cfg, output_dir):
    logging.info(f"\n========== STARTING RUN {run}/{cfg.runs} ==========")
    
    # Environment initialization
    env_run = gym.make("LunarLander-v3")
    state_dim = env_run.observation_space.shape[0]
    action_dim = env_run.action_space.n
    
    # Agent initialization
    if cfg.algorithm == "softmax_actor_critic":
        # Resolve actor/critic specific learning rates or fallback to base lr
        lr_actor = getattr(cfg, 'lr_actor', cfg.lr)
        lr_critic = getattr(cfg, 'lr_critic', cfg.lr)
        
        agent = ActorCriticAgent(
            state_dim, 
            action_dim, 
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=cfg.gamma,
            tau=cfg.tau,
            hidden_size=cfg.hidden_size,
            loss=cfg.loss,
            temp_init=cfg.temp_init,
            temp_min=cfg.temp_min,
            temp_decay=cfg.temp_decay,
            entropy_coef=getattr(cfg, 'entropy_coef', 0.0),
            use_average_reward=getattr(cfg, 'use_average_reward', False),
            avg_reward_alpha=getattr(cfg, 'avg_reward_alpha', 0.01)
        )
    else:
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
            temp_decay=cfg.temp_decay
        )
    
    # Training loop with periodic evaluation
    rewards_list, eval_history = train_agent(
        env=env_run,
        agent=agent,
        episodes=cfg.episodes,
        window_width=cfg.window_width,
        update_period=cfg.update_period,
        eval_period=cfg.eval_period,
        eval_runs=cfg.eval_runs,
        run_index=run,
        train_runs=cfg.runs
    )

    env_run.close()
    
    # Save model for current run
    run_model_path = os.path.join(output_dir, f"model_run{run}.pth")
    if cfg.algorithm == "softmax_actor_critic":
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict()
        }, run_model_path)
    else:
        torch.save(agent.q_network.state_dict(), run_model_path)
    logging.info(f"Model run {run} saved in {run_model_path}.")
        
    return rewards_list, eval_history

def main():
    # Reproducibility
    set_seed(42)

    # Arguments from CLI and YAML config loading
    args = argparse.ArgumentParser(description="Deep RL Agent for LunarLander using PyTorch")
    args.add_argument("--from-config", type=str, default=os.path.join("config", "config.yaml"), help="YAML configuration file")
    args.add_argument("--algorithm", type=str, choices=["q_learning", "sarsa", "expected_sarsa", "softmax_actor_critic"], help="Overrides algorithm from config")
    args.add_argument("--double", action='store_true', help="Use Double Q-Learning (or Double SARSA/Expected SARSA) if set")
    args.add_argument("--episodes", type=int, help="Number of training episodes")
    args.add_argument("--lr", type=float, help="Learning rate")
    args.add_argument("--lr-actor", type=float, help="Learning rate (Actor-Critic only)")
    args.add_argument("--lr-critic", type=float, help="Learning rate (Actor-Critic only)")
    args.add_argument("--tau", type=float, help="Soft update coefficient for target network")
    args.add_argument("--temp-decay", type=float, help="Temperature decay rate")
    args.add_argument("--temp-min", type=float, help="Minimum temperature for exploration")
    args.add_argument("--entropy-coef", type=float, help="Coefficient for entropy regularization")
    args.add_argument("--use-average-reward", action='store_true', help="Use average reward formulation instead of discounted")
    args.add_argument("--avg-reward-alpha", type=float, help="Step size for average reward update")
    args.add_argument("--eval-period", type=int, help="Frequency of evaluation (in episodes)")
    args.add_argument("--eval-runs", type=int, help="Number of evaluation episodes per evaluation phase")
    args.add_argument("--runs", type=int, help="Number of independent runs for averaging results")
    args.add_argument("--hidden-size", type=int, help="Number of hidden units in the Q-network")
    args.add_argument("--batch-size", type=int, help="Batch size for training")
    args.add_argument("--buffer-capacity", type=int, help="Capacity of the replay buffer")
    args.add_argument("--loss", type=str, choices=["mse", "smooth_l1"], help="Loss function for training")
    args.add_argument("--update-period", type=int, help="Frequency of agent learning updates (in steps)")
    args.add_argument("--output-dir", type=str, help="Output directory")
    parsed_args = args.parse_args()
    
    with open(parsed_args.from_config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        
    cfg = argparse.Namespace(**cfg_dict)
    set_default_config(cfg, parsed_args)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.output_dir, cfg.algorithm, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    setup_logger(output_dir)
    
    logging.info(f"Starting training with algorithm: '{cfg.algorithm}'")
    
    print(f"Configuration:\n{json.dumps(vars(cfg), indent=4)}")
    
    # Auto-save parameters at run start
    config_data = vars(cfg)
    config_path = os.path.join(output_dir, f"config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
        
    logging.info(f"Execution configuration saved in {config_path}")
    
    # Main loop over multiple runs
    rewards, eval_means, eval_episodes = [], [], []
    for run in range(1, cfg.runs + 1):
        rewards_list, eval_history = single_run(run, cfg, output_dir)
        
        rewards.append(rewards_list)
        eval_means.append([x[1] for x in eval_history])
        eval_episodes.append([x[0] for x in eval_history])

    # Aggregation
    rewards = np.array(rewards)
    mean_train_rewards = np.mean(rewards, axis=0)
    std_train_rewards = np.std(rewards, axis=0)
    
    # Save and plot results
    save_results(rewards, mean_train_rewards, std_train_rewards, eval_means, eval_episodes, output_dir)
    plot_training_curve(mean_train_rewards, std_train_rewards, eval_means, eval_episodes, cfg, output_dir)
    
    logging.info(f"\nTraining completed for {cfg.algorithm} with {cfg.runs} runs")
    
if __name__ == "__main__":
    main()
