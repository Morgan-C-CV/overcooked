import glob
import time
import os
import json
from datetime import datetime
import numpy as np
import keyboard
import threading
from queue import Queue
import torch
from environment.Overcooked import Overcooked_multi
from ray import tune
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
)
from ray.rllib.utils.numpy import convert_to_numpy, softmax
from ray.rllib.core.columns import Columns

class HumanGameplay:
    def __init__(self, ai_model_path, threshold=0.4):
        self.action_mapping = {
            'w': 3,  # up
            'a': 2,  # left
            's': 1,  # down
            'd': 0,  # right
            'p': 4   # stay/interact
        }
        
        # 游戏环境设置
        self.reward_config = {
            "metatask failed": -10,
            "goodtask finished": 20,
            "subtask finished": 30,
            "correct delivery": 200,
            "wrong delivery": -100,
            "step penalty": -0.5,
        }
        
        self.tasks = ["lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad"]
        self.env_params = {
            "grid_dim": [5, 5],
            "task": self.tasks[2],  # lettuce-onion salad
            "rewardList": self.reward_config,
            "map_type": "A",
            "mode": "vector",
            "debug": False,
            "possible_tasks": None,
            "max_steps": 400,
        }
        
        self.env = Overcooked_multi(**self.env_params)
        self.threshold = threshold
        self.ai_model = self.load_ai_model(ai_model_path)

        self.game_data = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'threshold': threshold,
            'task': self.env_params['task'],
            'trajectories': {'human': [], 'ai': []},
            'actions': {'human': [], 'ai': []},
            'rewards': {'human': [], 'ai': []},
            'cumulative_rewards': {'human': 0, 'ai': 0},
            'total_steps': 0,
            'game_duration': 0
        }

        self.input_queue = Queue()
        self.current_action = 4
        self.game_running = False

    def load_ai_model(self, model_path):
        print(f"Loading AI model from {model_path}...")

        checkpoint_dir = os.path.join(model_path, "PPO_Overcooked_63bb5_00000_0_2025-05-29_05-48-24")
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_000149")
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found at {checkpoint_path}")

        return RLModule.from_checkpoint(os.path.join(
            checkpoint_path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,
            'ai',
        ))

    def sample_ai_action(self, obs):
        mdl_out = self.ai_model.forward_inference({Columns.OBS: obs})
        if Columns.ACTION_DIST_INPUTS in mdl_out:
            logits = convert_to_numpy(mdl_out[Columns.ACTION_DIST_INPUTS])
            action = np.random.choice(list(range(len(logits[0]))), p=softmax(logits[0]))
            return action
        elif Columns.ACTIONS in mdl_out:
            return mdl_out[Columns.ACTIONS][0]
        else:
            raise NotImplementedError("Unexpected model output format")

    def input_thread(self):
        def on_press(event):
            if self.game_running and event.name in self.action_mapping:
                self.input_queue.put(self.action_mapping[event.name])

        for key in self.action_mapping:
            keyboard.on_press_key(key, lambda e: on_press(e))

        while self.game_running:
            time.sleep(0.01)

    def save_game_data(self):
        os.makedirs('logs', exist_ok=True)

        filename = f"logs/game_log_{self.game_data['timestamp']}.json"

        data_to_save = self.game_data.copy()
        for agent in ['human', 'ai']:
            data_to_save['trajectories'][agent] = [pos.tolist() if isinstance(pos, np.ndarray) else pos 
                                                 for pos in self.game_data['trajectories'][agent]]

        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"\nGame data saved to {filename}")

    def run_game(self):
        self.game_running = True
        self.env.game.on_init()
        obs, info = self.env.reset()

        input_thread = threading.Thread(target=self.input_thread)
        input_thread.start()
        
        start_time = time.time()
        last_step_time = start_time
        
        print("\n========== Game Started ============")
        print(f"Task: {self.env_params['task']}\n\n")
        print("Controls: WASD to move, P to interact")
        print("Press Ctrl+C to end the game")
        
        try:
            while self.game_running:
                current_time = time.time()

                if current_time - last_step_time >= 0.5:
                    try:
                        if not self.input_queue.empty():
                            while not self.input_queue.empty():
                                self.current_action = self.input_queue.get_nowait()
                        else:
                            self.current_action = 4
                    except:
                        self.current_action = 4

                    human_action = self.current_action
                    ai_action = self.sample_ai_action(torch.from_numpy(obs['ai']).unsqueeze(0).float())
                    actions = {'human': human_action, 'ai': ai_action}

                    new_obs, rewards, terminateds, _, info = self.env.step(actions)

                    for agent in ['human', 'ai']:
                        self.game_data['trajectories'][agent].append(obs[agent][:2])
                        self.game_data['actions'][agent].append(actions[agent])
                        self.game_data['rewards'][agent].append(rewards[agent])
                        self.game_data['cumulative_rewards'][agent] += rewards[agent]
                    
                    self.game_data['total_steps'] += 1

                    # print(f"\nStep {self.game_data['total_steps']}:")
                    action_name = [k for k, v in self.action_mapping.items() if v == human_action][0]
                    # print(f"Human action: {action_name} ({human_action})")
                    # print(f"AI action: {ai_action}")
                    # print(f"Rewards - Human: {rewards['human']:.1f}, AI: {rewards['ai']:.1f}")
                    # print(f"Cumulative - Human: {self.game_data['cumulative_rewards']['human']:.1f}, "
                    #       f"AI: {self.game_data['cumulative_rewards']['ai']:.1f}")
                    
                    if 'submitted_dish' in info:
                        print("\n=== Delivery Information ===")
                        print(f"Submitted dish: {info['submitted_dish']}")
                        print(f"Expected dish: {self.env_params['task']}")
                        print(f"Correct? {info['submitted_dish'] == self.env_params['task']}")

                    obs = new_obs
                    last_step_time = current_time
                    self.env.render()

                    if terminateds['__all__']:
                        print("\n=== Game Over ===")
                        break
                    
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n=== Game Interrupted ===")

        self.game_running = False
        input_thread.join()
        self.game_data['game_duration'] = time.time() - start_time
        self.save_game_data()

        print("\n=== Game Summary ===")
        print(f"Duration: {self.game_data['game_duration']:.1f} seconds")
        print(f"Total steps: {self.game_data['total_steps']}")
        print(f"Final score - Human: {self.game_data['cumulative_rewards']['human']:.1f}, "
              f"AI: {self.game_data['cumulative_rewards']['ai']:.1f}")

def main():
    model_path = "c:/Users/16146/PycharmProjects/overcooked/runs/collaborative_training_threshold_0.4_1748486904234"
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path not found: {model_path}")

    game = HumanGameplay(model_path, threshold=0.4)
    game.run_game()

if __name__ == "__main__":
    main() 