import argparse
import copy


from environment.Overcooked import Overcooked_multi
from Agents import *
import pandas as pd

# TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]

TASKLIST = ["lettuce-tomato salad", "onion-tomato salad", "lettuce-onion-tomato salad"]

class Player:
    ACTION_MAPPING = {
        "w": 3,
        "d": 0,
        "a": 2,
        "s": 1,
        "q": 4
    }

    REWARD_LIST = {
        "subtask finished": 10,
        "correct delivery": 200,
        "goodtask finished": 10,
        "wrong delivery": -100,
        "step penalty": -0.1,
        "metatask failed": -10,
    }

    def __init__(self, env_id, grid_dim, task, map_type, n_agent, obs_radius, mode, debug, agent='human'):
        self.env_params = {
            'grid_dim': grid_dim,
            'task': TASKLIST[2],
            'rewardList': self.REWARD_LIST,
            'map_type': map_type,
            'mode': mode,
            'debug': debug
        }
        self.env = Overcooked_multi(**self.env_params)

        if agent == 'stationary':

            self.agent = AlwaysStationaryRLM(
                observation_space=self.env.observation_spaces['ai'],
                action_space=self.env.action_spaces['ai'],
                inference_only=True
            )

        elif agent == 'random':
            self.agent = RandomRLM(
                observation_space=self.env.observation_spaces['ai'],
                action_space=self.env.action_spaces['ai'],
                inference_only=True
            )

        elif agent == 'human':
            self.agent = 'human'

        else:
            raise NotImplementedError(f'{agent} is unknonw')


        self.rewards = 0
        self.discount = 1
        self.step = 0

    def run(self):
        self.env.game.on_init()
        new_obs, _ = self.env.reset()
        self.env.render()
        data = [["obs", "action_human", "action_ai", "new_obs", "reward_human", "reward_ai", "done"]]

        total_rewards = 0
        episode_steps = 0
        completed_tasks = 0
        total_steps = 0
        episodes = 0
        
        while True:
            obs = new_obs
            row = [obs['human']]
            self.step += 1
            episode_steps += 1
            total_steps += 1
            
            input_human = input("Input Human: ").strip().split(" ")

            if input_human == ['p']:
                self.save_data(data)
                print("\n=== Current Statistics ===")
                print(f"Episodes completed: {episodes}")
                print(f"Total steps: {total_steps}")
                print(f"Average steps per episode: {total_steps/(episodes+1):.2f}")
                print(f"Total rewards: {total_rewards:.2f}")
                print(f"Average reward per episode: {total_rewards/(episodes+1):.2f}")
                print(f"Tasks completed: {completed_tasks}")
                print("========================\n")
                continue

            if self.agent == 'human':
                input_ai = input("Input AI: ").strip().split(" ")
            else:
                input_ai = self.agent._forward_inference({"obs": [obs['ai']]})['actions']

            action = {
                "human": self.ACTION_MAPPING[input_human[0]],
                "ai": self.ACTION_MAPPING[input_ai[0]]
            }

            row.append(action['human'])
            row.append(action['ai'])

            new_obs, reward, done, _, _ = self.env.step(action)

            total_rewards += reward['human']
            
            row.append(new_obs['human'])
            row.append(reward['human'])
            row.append(reward['ai'])
            row.append(done['__all__'])

            data.append(copy.deepcopy(row))

            self.env.render()

            if done['__all__']:
                episodes += 1
                if all(status == 0 for status in self.env.taskCompletionStatus):
                    completed_tasks += 1

                print("\n=== Episode {} Complete ===".format(episodes))
                print(f"Episode steps: {episode_steps}")
                print(f"Episode reward: {reward['human']:.2f}")
                print(f"Task completed: {'Yes' if all(status == 0 for status in self.env.taskCompletionStatus) else 'No'}")
                print("========================\n")

                episode_steps = 0

                self.save_data(data)
                print("\n=== Final Statistics ===")
                print(f"Total episodes: {episodes}")
                print(f"Total steps: {total_steps}")
                print(f"Average steps per episode: {total_steps/episodes:.2f}")
                print(f"Total rewards: {total_rewards:.2f}")
                print(f"Average reward per episode: {total_rewards/episodes:.2f}")
                print(f"Tasks completed: {completed_tasks}")
                print(f"Task completion rate: {(completed_tasks/episodes)*100:.2f}%")
                print("======================\n")
                break

    def save_data(self, data):
        columns = data[0]
        data = data[1:]
        df = pd.DataFrame(data, columns=columns)
        csv_filename = "output.csv"
        df.to_csv(csv_filename, index=False)

        with open(csv_filename, 'a') as f:
            f.write("\n=== Statistics ===\n")
            f.write(f"Total steps: {self.step}\n")
            f.write(f"Total rewards: {self.rewards:.2f}\n")
            f.write(f"Average reward per step: {self.rewards/self.step:.2f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_dim', type=int, nargs=5, default=[5, 5], help='Grid world size')
    parser.add_argument('--task', type=int, default=1, help='The recipe agent cooks')
    parser.add_argument('--map_type', type=str, default="A", help='The type of map')
    parser.add_argument('--mode', type=str, default="vector", help='The type of observation (vector/image)')
    parser.add_argument('--debug', type=bool, default=True, help='Whether to print debug information and render')

    params = vars(parser.parse_args())

    params['env_id'] = "default_env_id"
    params['n_agent'] = 2
    params['obs_radius'] = 5

    player = Player(**params)
    player.run()