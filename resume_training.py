import time
import ray
from ray.train import RunConfig, CheckpointConfig
from environment.Overcooked import Overcooked_multi
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from Agents import AlwaysStationaryRLM, RandomRLM
import os

def define_env():
    reward_config = {
        "metatask failed": -5,
        "goodtask finished": 15,
        "subtask finished": 25,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -1.0,
    }

    tasks = [
        "tomato salad", 
        "lettuce salad", 
        "onion salad",
        "lettuce-tomato salad",
        "onion-tomato salad",
        "lettuce-onion salad",
        "lettuce-onion-tomato salad"
    ]
    
    def env_creator(env_context):
        worker_index = env_context.worker_index if hasattr(env_context, 'worker_index') else 0
        task_index = worker_index % len(tasks)
        
        env_params = {
            "grid_dim": [5, 5],
            "task": tasks[task_index],
            "rewardList": reward_config,
            "map_type": "A",
            "mode": "vector",
            "debug": False,
            "possible_tasks": tasks,
            "max_steps": 400,
        }
        return Overcooked_multi(**env_params)

    register_env(
        "Overcooked",
        env_creator,
    )

def define_agents():
    human_policy = RLModuleSpec(module_class=AlwaysStationaryRLM)
    policies_to_train = ['ai']
    return human_policy, policies_to_train

def define_training(human_policy, policies_to_train):
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("Overcooked")
        .env_runners(
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0.5
        )
        .multi_agent(
            policies={"ai", "human"},
            policy_mapping_fn=lambda aid, *a, **kw: aid,
            policies_to_train=policies_to_train
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "human": human_policy,
                    "ai": RLModuleSpec(),
                }
            ),
        )
        .training(
            lr=1e-4,
            lambda_=0.95,
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff=0.2,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            num_epochs=8,
            minibatch_size=256,
            train_batch_size=4000,
        )
        .framework("torch")
    )
    return config

def resume_training(checkpoint_path, additional_iterations=1000):
    ray.init(num_gpus=1)

    define_env()
    human_policy, policies_to_train = define_agents()
    config = define_training(human_policy, policies_to_train)

    experiment_dir = os.path.dirname(checkpoint_path)
    
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, "runs")
    experiment_name = f"resume_training_{int(time.time() * 1000)}"

    checkpoint_epochs = [300, 400, 800, 1000]
    
    config = config.to_dict()
    config["resume"] = True
    config["resume_from_checkpoint"] = checkpoint_path
    config["checkpoint_at_end"] = True
    config["keep_checkpoints_num"] = len(checkpoint_epochs) + 1
    
    checkpoint_config = CheckpointConfig(
        checkpoint_frequency=None,
        checkpoint_at_end=True,
        num_to_keep=len(checkpoint_epochs) + 1, 
        checkpoint_score_attribute="episode_reward_mean",
        checkpoint_score_order="max",
    )

    run_config = RunConfig(
        storage_path=storage_path,
        name=experiment_name,
        stop={"training_iteration": additional_iterations},
        checkpoint_config=checkpoint_config,
        callbacks=[
            tune.callbacks.CheckpointCallback(
                checkpoint_epochs,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max"
            )
        ]
    )

    tuner = tune.Tuner(
        trainable="PPO",
        param_space=config,
        run_config=run_config
    )
    
    tuner.fit()

if __name__ == "__main__":
    base_dir = "c:\\Users\\16146\\PycharmProjects\\overcooked\\runs\\run_stationary_1747690505862\\PPO_Overcooked_21573_00000_0_2025-05-20_00-35-06"
    checkpoint_path = os.path.join(base_dir, "checkpoint_000074")
    iterations = 1000
    resume_training(checkpoint_path, iterations)