import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

def set_seed(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def setup_logger(cfg, timestamp):
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{output_dir}/training_{timestamp}.log",
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    if len(ret) >= n:
        ret[n:] = ret[n:] - ret[:-n]
        res = ret / n
        res[:n-1] = ret[:n-1] / np.arange(1, n)
        return res
    return ret / np.arange(1, len(ret) + 1)

def set_default_config(cfg, parsed_args):
    """
    Overwrites YAML config parameters with command-line arguments (if present).
    Priority: argparse > YAML config.
    """
    for key, value in vars(parsed_args).items():
        if value is not None:
            setattr(cfg, key, value)
            
def plot_training_curve(mean_train_rewards, std_train_rewards, all_runs_eval_means, eval_eps, cfg, timestamp):
    plt.figure(figsize=(10,6))
    plt.plot(mean_train_rewards, label="Training Reward", alpha=0.3, color='blue')
    
    if cfg.runs > 1:
        plt.fill_between(range(cfg.episodes), mean_train_rewards - std_train_rewards, mean_train_rewards + std_train_rewards, alpha=0.1, color='blue')
    
    ma_train = moving_average(mean_train_rewards, n=cfg.window_width)
    plt.plot(ma_train, label=f"Training Reward (smoothed)", color='red')
    
    if len(all_runs_eval_means) > 0:
        all_runs_eval_means = np.array(all_runs_eval_means)
        global_eval_means = np.mean(all_runs_eval_means, axis=0)
        global_eval_stds = np.std(all_runs_eval_means, axis=0)
        
        # eval_eps is a list of lists (one per run), so we only need the first one for the X-axis
        x_eval = eval_eps[0] if len(eval_eps) > 0 and isinstance(eval_eps[0], list) else eval_eps

        plt.errorbar(
            x_eval, global_eval_means, yerr=global_eval_stds, 
            fmt='-o', color='gold', ecolor='darkorange', linewidth=2, capsize=4,
            label=f'Mean Evaluation Reward'
        )
    
    plt.axhline(200, color='green', linestyle='--', label='Solution Threshold')
    plt.xlabel('Episodes')
    plt.ylabel(f'Total reward (averaged over {cfg.runs} runs)')
    algorithm_name = " ".join(word.capitalize() for word in cfg.algorithm.split("_"))
    plt.title(f'Learning curve - {algorithm_name} (mean of {cfg.runs} runs)' if cfg.runs > 1 else f'Learning curve - {algorithm_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plot_path = os.path.join(cfg.output_dir, f"reward_plot_{cfg.algorithm}_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    
    logging.info(f"Plot saved in: {plot_path}")