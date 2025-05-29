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
            "possible_tasks": None,  # Set to None to prevent random task selection
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
    # 计算entropy coefficient的衰减
    def entropy_schedule(train_iter):
        # 从0.3开始，在1500轮内线性衰减到0.1
        start_entropy = 0.3
        end_entropy = 0.1
        decay_steps = 1500
        current_entropy = max(
            end_entropy,
            start_entropy - (start_entropy - end_entropy) * (train_iter / decay_steps)
        )
        return current_entropy

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("Overcooked")
        .env_runners(
            num_envs_per_env_runner=2,  # 增加环境数量以提高探索
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
            lr=3e-4,  # 增加学习率
            lambda_=0.95,
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff=0.3,  # 增加初始熵以促进探索
            entropy_coeff_schedule=entropy_schedule,  # 添加熵系数衰减
            vf_loss_coeff=1.0,  # 增加价值函数损失权重
            grad_clip=1.0,  # 增加梯度裁剪范围
            num_epochs=10,  # 增加每批次训练轮数
            minibatch_size=256,
            train_batch_size=8000,  # 增加批次大小以提高稳定性
            # 添加KL散度目标以防止策略剧烈变化
            kl_coeff=0.2,
            kl_target=0.01,
        )
        .framework("torch")
        .rollouts(num_rollout_workers=2)  # 增加rollout workers数量
    )
    return config

def resume_training(checkpoint_path, additional_iterations=1500):  # 增加训练轮数
    ray.init(num_gpus=1)

    define_env()
    human_policy, policies_to_train = define_agents()
    config = define_training(human_policy, policies_to_train)

    experiment_dir = os.path.dirname(checkpoint_path)
    
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, "runs")
    experiment_name = f"resume_training_{int(time.time() * 800)}"

    checkpoint_epochs = [300, 600, 900, 1200, 1500]  # 更多的检查点
    
    config = config.to_dict()
    config["resume"] = True
    config["resume_from_checkpoint"] = checkpoint_path
    config["checkpoint_at_end"] = True
    config["keep_checkpoints_num"] = len(checkpoint_epochs) + 1
    
    checkpoint_config = CheckpointConfig(
        checkpoint_frequency=300,  # 每300轮保存一次
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
    )

    tuner = tune.Tuner(
        trainable="PPO",
        param_space=config,
        run_config=run_config
    )
    
    tuner.fit()

if __name__ == "__main__":
    base_dir = "c:\\Users\\16146\\PycharmProjects\\overcooked\\runs\\run_stationary_1748332915148\\PPO_Overcooked_db1e4_00000_0_2025-05-27_11-01-55"
    checkpoint_path = os.path.join(base_dir, "checkpoint_000059")
    iterations = 1500  # 增加训练轮数到1500
    resume_training(checkpoint_path, iterations)