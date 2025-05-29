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
            "task": tasks[2],
            "rewardList": reward_config,
            "map_type": "A",
            "mode": "vector",
            "debug": False,
            "possible_tasks": None,
            "max_steps": 400,
        }

env = Overcooked_multi(**env_params)


def sample_action(mdl, obs):
    mdl_out = mdl.forward_inference({Columns.OBS: obs})
    if Columns.ACTION_DIST_INPUTS in mdl_out: #our custom policies might return the actions directly, while learned policies might return logits.
        logits = convert_to_numpy(mdl_out[Columns.ACTION_DIST_INPUTS])
        action = np.random.choice(list(range(len(logits[0]))), p=softmax(logits[0]))
        return action
    elif 'actions' in mdl_out:
        return mdl_out['actions'][0]

    else:
        raise NotImplementedError("Something weird is going on when sampling acitons")

def load_modules(args):
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, args.save_dir)
    p = f"{storage_path}/{args.name}_{args.rl_module}_*"
    experiment_name = glob.glob(p)[-1]
    print(f"Loading results from {experiment_name}...")
    restored_tuner = tune.Tuner.restore(experiment_name, trainable="PPO")
    result_grid = restored_tuner.get_results()
    best_result = result_grid.get_best_result(metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max")
    print(best_result.config)
    best_checkpoint = best_result.checkpoint
    human_module = RLModule.from_checkpoint(os.path.join(
        best_checkpoint.path,
        COMPONENT_LEARNER_GROUP,
        COMPONENT_LEARNER,
        COMPONENT_RL_MODULE,
        'human',
    ))
    ai_module = RLModule.from_checkpoint(os.path.join(
        best_checkpoint.path,
        COMPONENT_LEARNER_GROUP,
        COMPONENT_LEARNER,
        COMPONENT_RL_MODULE,
        'ai',
    ))
    return ai_module, human_module


def main(args):
    ai_module, human_module = load_modules(args)
    env.game.on_init()
    obs, info = env.reset()
    
    # Initialize cumulative rewards
    cumulative_rewards = {'human': 0, 'ai': 0}
    step_count = 0

    print("\n=== Starting Episode ===")
    env.render()

    while True:
        ai_action = sample_action(ai_module, torch.from_numpy(obs['ai']).unsqueeze(0).float())
        human_action = sample_action(human_module, torch.from_numpy(obs['human']).unsqueeze(0).float())
        actions = {'human': human_action,
                   'ai': ai_action}

        obs, rewards, terminateds, _, info = env.step(actions)
        env.render()
        
        # Update cumulative rewards
        for agent in rewards:
            cumulative_rewards[agent] += rewards[agent]
        step_count += 1
        
        print(f"\nStep {step_count}:")
        print(f"Step rewards: {rewards}")
        print(f"Cumulative rewards: {cumulative_rewards}")
        
        # Print detailed info about submissions and rewards
        if 'submitted_dish' in info:
            print("\n=== Delivery Information ===")
            print(f"提交的菜品 (Submitted dish): {info['submitted_dish']}")
            print(f"预期的菜品 (Expected dish): {env_params['task']}")
            print(f"是否正确 (Correct?): {info['submitted_dish'] == env_params['task']}")
            
            # Check if rewards match the delivery outcome
            expected_delivery_reward = reward_config['correct delivery'] if info['submitted_dish'] == env_params['task'] else reward_config['wrong delivery']
            actual_delivery_reward = rewards['human']  # Assuming both agents get same reward
            print(f"预期投递奖励 (Expected delivery reward): {expected_delivery_reward}")
            print(f"实际获得奖励 (Actual reward): {actual_delivery_reward}")
            
        time.sleep(0.1)

        if terminateds['__all__']:
            print("\n=== Episode Summary ===")
            print(f"Total steps: {step_count}")
            print(f"Final cumulative rewards: {cumulative_rewards}")
            print(f"Expected step penalty total: {step_count * reward_config['step penalty']}")
            break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="runs", type=str)
    parser.add_argument("--name", default="run", type=str)
    parser.add_argument("--rl_module", default="stationary", type=str)

    args = parser.parse_args()
    main(args)
