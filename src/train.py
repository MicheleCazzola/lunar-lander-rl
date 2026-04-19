import logging
import numpy as np

def evaluate_agent(env, agent, eval_runs=30):
    """
    Executes a pure test phase: the agent plays without exploration (greedy)
    and without updating networks or saving anything in the Replay Buffer.
    """
    eval_rewards = []
    for _ in range(eval_runs):
        state, _ = env.reset()
        done, truncated = False, False
        tot_rew = 0
        while not (done or truncated):
            # Greedy action selection for evaluation (no exploration)
            action = agent.act(state, eval_mode=True)
            state, reward, done, truncated, _ = env.step(action)
            tot_rew += reward
        eval_rewards.append(tot_rew)
    
    return np.mean(eval_rewards), np.std(eval_rewards)

def train_agent(env, agent, episodes, window_width, update_period, eval_period, eval_runs, run_index, train_runs):
    """
    Executes the core training loop and invokes periodic evaluations.
    """
    rewards_list = []
    eval_history = []  # Stores tuples (episode, mean_reward, std_reward)

    for e in range(episodes):
        state, info = env.reset()
        action = agent.act(state) # Chooses first action (for pure Sarsa)
        
        total_reward = 0
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated):
            # Interaction
            next_state, reward, done, truncated, info = env.step(action)
            next_action = agent.act(next_state) if not (done or truncated) else None
            
            # Save into buffer
            agent.memory.append(state, action, reward, next_state, (done or truncated), next_action)
            
            if (step_count + 1) % update_period == 0:
                # Explicit learning on a mini-batch (Actual Training)
                agent.learn()

            state = next_state
            action = next_action 
            total_reward += reward
            step_count += 1

        # Temperature decay at episode end
        agent.update_temp()
        
        rewards_list.append(total_reward)

        # Compute Moving Average
        avg_reward = np.mean(rewards_list[-window_width:]) if len(rewards_list) >= window_width else np.mean(rewards_list)
        
        log_msg = f"Run: {run_index:>2d}/{train_runs} | Episode: {e+1:>3d}/{episodes} | Temp.: {agent.temp:.3f} | Reward: {total_reward: 7.2f} | Mean reward: {avg_reward: 7.2f}"
        logging.info(log_msg)
        
        # ======= EVALUATION PHASE =======
        # Evaluate with greedy policy (no exploration)
        if (e + 1) % eval_period == 0:
            mean_eval, std_eval = evaluate_agent(env, agent, eval_runs)
            eval_history.append((e+1, mean_eval, std_eval))
            
            eval_log_msg = f"*** EVALUATION | Episode: {e+1} | Mean Reward: {mean_eval:7.2f} ± {std_eval:5.2f} ***"
            logging.info(eval_log_msg)

    return rewards_list, eval_history