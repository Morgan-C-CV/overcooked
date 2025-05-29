import time
import ray
from ray.train import RunConfig, CheckpointConfig
from environment.Overcooked import Overcooked_multi
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec, RLModule
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from Agents import AlwaysStationaryRLM, RandomRLM, HybridRandomRLM
import os
import glob

# Configuration
TRAINING_CONFIG = {
    "base_dir": "c:/Users/16146/PycharmProjects/overcooked/runs/run_stationary_1748332915148/PPO_Overcooked_db1e4_00000_0_2025-05-27_11-01-55",
    "checkpoint": "checkpoint_000059",
    "thresholds": [0.5,0.3],
    "iterations": 1500,
    "save_dir": "runs",
    "name": "collaborative_training",
    "save_best_only": True,
    "best_reward_threshold": 270
}

def define_env():
    reward_config = {
        "metatask failed": -10,
        "goodtask finished": 20,
        "subtask finished": 30,
        "correct delivery": 200,
        "wrong delivery": -100,
        "step penalty": -0.5,
    }

    tasks = ["lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad"]
    
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
            "possible_tasks": None,
            "max_steps": 400,
        }
        return Overcooked_multi(**env_params)

    register_env(
        "Overcooked",
        env_creator,
    )

def load_human_model(model_path):
    """Load the trained human model from checkpoint."""
    base_module = RLModule.from_checkpoint(os.path.join(
        model_path,
        'learner_group',
        'learner',
        'rl_module',
        'human'
    ))
    return base_module

def load_ai_model(model_path):
    """Load the trained AI model from checkpoint."""
    ai_module = RLModule.from_checkpoint(os.path.join(
        model_path,
        'learner_group',
        'learner',
        'rl_module',
        'ai'
    ))
    return ai_module

def define_agents(base_dir, checkpoint, threshold):
    # Construct full paths for both models
    checkpoint_path = os.path.join(base_dir, checkpoint)
    
    # Load the trained human model
    base_module = load_human_model(checkpoint_path)
    
    # Create hybrid policy with specified threshold
    human_policy = RLModuleSpec(
        module_class=HybridRandomRLM,
        observation_space=base_module.observation_space,
        action_space=base_module.action_space,
        model_config={'base_module': base_module, 'threshold': threshold}
    )

    # Load the trained AI model
    ai_module = load_ai_model(checkpoint_path)
    ai_policy = RLModuleSpec(
        module_class=type(ai_module),
        observation_space=ai_module.observation_space,
        action_space=ai_module.action_space,
        model_config={'base_module': ai_module}
    )

    policies_to_train = ['ai']
    return human_policy, ai_policy, policies_to_train

def define_training(human_policy, ai_policy, policies_to_train):
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
                    "ai": ai_policy,
                }
            ),
        )
        .training(
            lr=5e-4,
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

def train_single_threshold(config, threshold):
    """Train model with a single threshold value."""
    print(f"\n{'='*50}")
    print(f"Starting training with threshold: {threshold}")
    print(f"{'='*50}\n")
    
    ray.init(num_gpus=1)
    
    define_env()
    human_policy, ai_policy, policies_to_train = define_agents(
        config["base_dir"],
        config["checkpoint"],
        threshold
    )
    training_config = define_training(human_policy, ai_policy, policies_to_train)
    
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, config["save_dir"])
    experiment_name = f"{config['name']}_threshold_{threshold}_{int(time.time() * 1000)}"
    
    best_reward = -float('inf')
    best_checkpoint = None
    
    def custom_stopper(trial_id, result):
        nonlocal best_reward, best_checkpoint
        current_reward = result.get("episode_reward_mean", 0)
        
        if current_reward > best_reward:
            best_reward = current_reward
            best_checkpoint = result.get("checkpoint")
            
            if current_reward > config["best_reward_threshold"]:
                save_path = os.path.join(storage_path, f"{experiment_name}_best")
                print(f"\nSaving best model with reward {current_reward} to {save_path}")
                os.makedirs(save_path, exist_ok=True)
                if best_checkpoint:
                    import shutil
                    shutil.copytree(best_checkpoint, os.path.join(save_path, "checkpoint"), dirs_exist_ok=True)
        
        return result["training_iteration"] >= config["iterations"]
    
    tuner = tune.Tuner(
        "PPO",
        param_space=training_config,
        run_config=RunConfig(
            storage_path=storage_path,
            name=experiment_name,
            stop=custom_stopper,
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
                num_to_keep=2
            ),
        )
    )
    tuner.fit()
    ray.shutdown()
    
    print(f"\nTraining completed for threshold {threshold}")
    print(f"Best reward achieved: {best_reward}")

def main():
    # Train for each threshold
    for threshold in TRAINING_CONFIG["thresholds"]:
        train_single_threshold(TRAINING_CONFIG, threshold)

if __name__ == "__main__":
    main()