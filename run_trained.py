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
    "metatask failed": -1,
    "goodtask finished": 12,
    "subtask finished": 25,
    "correct delivery": 200,
    "wrong delivery": -50,
    "step penalty": -0.5,
}
env_params = {
    "grid_dim": [7, 7],
    "task": "tomato salad",
    "rewardList": reward_config,
    "map_type": "A",
    "mode": "vector",
    "debug": False,
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
    env.render()

    while True:
        ai_action = sample_action(ai_module, torch.from_numpy(obs['ai']).unsqueeze(0).float())
        human_action = sample_action(human_module, torch.from_numpy(obs['human']).unsqueeze(0).float())
        actions = {'human': human_action,
                   'ai': ai_action}

        obs, rewards, terminateds, _, _ = env.step(actions)
        env.render()
        time.sleep(0.1)

        if terminateds['__all__']:
            break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="runs", type=str)
    parser.add_argument("--name", default="run", type=str)
    parser.add_argument("--rl_module", default="stationary", type=str)

    args = parser.parse_args()
    main(args)
