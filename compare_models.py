import glob
import time
from environment.Overcooked import Overcooked_multi
from ray import tune
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
)
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray.rllib.core.columns import Columns
import torch
import os
from ray.rllib.utils.numpy import convert_to_numpy, softmax
import numpy as np
from Agents import HybridRandomRLM

reward_config = {
        "metatask failed": -10,
        "goodtask finished": 20,
        "subtask finished": 30,
        "correct delivery": 200,
        "wrong delivery": -100,
        "step penalty": -0.5,
    }

tasks = ["lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad"]

env_params = {
            "grid_dim": [5, 5],
            "task": tasks[2],  # Explicitly set to lettuce-onion salad
            "rewardList": reward_config,
            "map_type": "A",
            "mode": "vector",
            "debug": True,  # Enable debug mode
            "possible_tasks": None,
            "max_steps": 400,
        }

env = Overcooked_multi(**env_params)

def sample_action(mdl, obs):
    mdl_out = mdl.forward_inference({Columns.OBS: obs})
    if isinstance(mdl, HybridRandomRLM):
        # HybridRandomRLM directly returns actions
        return mdl_out[Columns.ACTIONS][0]
    elif Columns.ACTION_DIST_INPUTS in mdl_out:
        logits = convert_to_numpy(mdl_out[Columns.ACTION_DIST_INPUTS])
        action = np.random.choice(list(range(len(logits[0]))), p=softmax(logits[0]))
        return action
    elif Columns.ACTIONS in mdl_out:
        return mdl_out[Columns.ACTIONS][0]
    else:
        raise NotImplementedError("Something weird is going on when sampling actions")

def load_module_from_path(experiment_path, is_human=False, threshold=None):
    print(f"Loading results from {experiment_path}...")
    restored_tuner = tune.Tuner.restore(experiment_path, trainable="PPO")
    result_grid = restored_tuner.get_results()
    best_result = result_grid.get_best_result(metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max")
    print(best_result.config)
    best_checkpoint = best_result.checkpoint
    
    # Load base module (always load AI model as it's the trained one)
    base_module = RLModule.from_checkpoint(os.path.join(
        best_checkpoint.path,
        COMPONENT_LEARNER_GROUP,
        COMPONENT_LEARNER,
        COMPONENT_RL_MODULE,
        'ai',
    ))
    
    # If loading human agent, wrap it in HybridRandomRLM
    if is_human:
        hybrid_model = HybridRandomRLM(
            observation_space=base_module.observation_space,
            action_space=base_module.action_space,
            model_config={'base_module': base_module, 'threshold': threshold or 1.0}
        )
        return hybrid_model
    return base_module

def get_experiment_path(base_dir, threshold=None):
    if threshold is not None:
        pattern = f"collaborative_training_threshold_{threshold}_*"
    else:
        pattern = "run_stationary_*"
    matching_dirs = glob.glob(os.path.join(base_dir, pattern))
    if not matching_dirs:
        raise ValueError(f"No matching directories found for pattern: {pattern}")
    return matching_dirs[-1]  # Return the latest matching directory

def analyze_agent_behavior(agent_name, action, obs, last_positions):
    """Analyze agent behavior and detect potential issues"""
    current_pos = obs[:2] if isinstance(obs, np.ndarray) else None
    analysis = {
        'repeated_action': False,
        'stuck': False,
        'out_of_bounds': False
    }
    
    if current_pos is not None:
        if last_positions[agent_name] is not None:
            if np.array_equal(current_pos, last_positions[agent_name]):
                analysis['stuck'] = True
            if any(pos < 0 or pos > 1 for pos in current_pos):
                analysis['out_of_bounds'] = True
    
    return analysis

def main(args):
    current_dir = os.getcwd()
    runs_dir = os.path.join(current_dir, "runs")
    
    # 加载模型
    stationary_path = get_experiment_path(runs_dir)
    human_module = load_module_from_path(stationary_path, is_human=True, threshold=0.4)
    
    collaborative_path = get_experiment_path(runs_dir, threshold=0.4)
    ai_module = load_module_from_path(collaborative_path)
    
    env.game.on_init()
    obs, info = env.reset()
    
    # Initialize tracking variables
    cumulative_rewards = {'human': 0, 'ai': 0}
    step_count = 0
    last_positions = {'human': None, 'ai': None}
    stuck_count = {'human': 0, 'ai': 0}
    action_history = {'human': [], 'ai': []}
    subtask_completion = {'chopped_lettuce': False, 'chopped_onion': False}

    print("\n=== Starting Episode ===")
    print(f"Human Agent: Using model from {stationary_path} with threshold 0.4")
    print(f"AI Agent: Using model from {collaborative_path}")
    print(f"Task: {env_params['task']}")
    print("\nInitial Environment State:")
    print(f"Observation Space: {obs['human'].shape}")
    env.render()

    while True:
        # Sample actions
        ai_action = sample_action(ai_module, torch.from_numpy(obs['ai']).unsqueeze(0).float())
        human_action = sample_action(human_module, torch.from_numpy(obs['human']).unsqueeze(0).float())
        actions = {'human': human_action, 'ai': ai_action}
        
        # Store current observations for analysis
        current_obs = obs.copy()
        
        # Take environment step
        obs, rewards, terminateds, _, info = env.step(actions)
        env.render()
        
        # Update tracking variables
        for agent in rewards:
            cumulative_rewards[agent] += rewards[agent]
            action_history[agent].append(actions[agent])
        step_count += 1
        
        # Print step information
        print(f"\nStep {step_count}:")
        print(f"Step rewards: {rewards}")
        print(f"Cumulative rewards: {cumulative_rewards}")
        print(f"Actions: AI={ai_action}, Human={human_action}")
        
        # Analyze agent behavior
        for agent in ['human', 'ai']:
            current_pos = obs[agent][:2] if isinstance(obs[agent], np.ndarray) else None
            behavior = analyze_agent_behavior(agent, actions[agent], obs[agent], last_positions)
            
            if behavior['stuck']:
                stuck_count[agent] += 1
                print(f"Warning: {agent} might be stuck at position {current_pos}")
            
            if behavior['out_of_bounds']:
                print(f"Warning: {agent} position {current_pos} may be out of bounds")
            
            # Update last known position
            last_positions[agent] = current_pos
        
        # Analyze task progress
        if 'submitted_dish' in info:
            print("\n=== Delivery Information ===")
            print(f"提交的菜品 (Submitted dish): {info['submitted_dish']}")
            print(f"预期的菜品 (Expected dish): {env_params['task']}")
            print(f"是否正确 (Correct?): {info['submitted_dish'] == env_params['task']}")
            
            expected_delivery_reward = reward_config['correct delivery'] if info['submitted_dish'] == env_params['task'] else reward_config['wrong delivery']
            actual_delivery_reward = rewards['human']
            print(f"预期投递奖励 (Expected delivery reward): {expected_delivery_reward}")
            print(f"实际获得奖励 (Actual reward): {actual_delivery_reward}")
        
        # Analyze negative rewards
        if any(r < -1 for r in rewards.values()):
            print("\n=== Negative Reward Analysis ===")
            print(f"Current positions - AI: {obs['ai'][:2]}, Human: {obs['human'][:2]}")
            print(f"Previous positions - AI: {last_positions['ai']}, Human: {last_positions['human']}")
            print(f"Stuck counts - AI: {stuck_count['ai']}, Human: {stuck_count['human']}")
            
            # Analyze action patterns
            for agent in ['human', 'ai']:
                if len(action_history[agent]) >= 5:
                    recent_actions = action_history[agent][-5:]
                    print(f"{agent.capitalize()} recent action pattern: {recent_actions}")
        
        time.sleep(0.1)

        if terminateds['__all__']:
            print("\n=== Episode Summary ===")
            print(f"Total steps: {step_count}")
            print(f"Final cumulative rewards: {cumulative_rewards}")
            print(f"Expected step penalty total: {step_count * reward_config['step penalty']}")
            print(f"Stuck episodes - AI: {stuck_count['ai']}, Human: {stuck_count['human']}")
            
            # Print action distribution
            for agent in ['human', 'ai']:
                action_counts = np.bincount(action_history[agent], minlength=5)
                action_dist = action_counts / len(action_history[agent])
                print(f"\n{agent.capitalize()} Action Distribution:")
                print(f"Right (0): {action_dist[0]:.2%}")
                print(f"Down  (1): {action_dist[1]:.2%}")
                print(f"Left  (2): {action_dist[2]:.2%}")
                print(f"Up    (3): {action_dist[3]:.2%}")
                print(f"Stay  (4): {action_dist[4]:.2%}")
            break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=1.0,
                      help="Threshold value for collaborative training model selection")
    parser.add_argument("--human_threshold", type=float, default=1.0,
                      help="Threshold value for human agent's hybrid policy")
    args = parser.parse_args()
    main(args) 