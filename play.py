import argparse
import copy


from environment.Overcooked import Overcooked_multi
from Agents import *
import pandas as pd

TASKLIST = [
    "tomato salad", "lettuce salad", "onion salad",
    "lettuce-tomato salad", "onion-tomato salad",
    "lettuce-onion salad", "lettuce-onion-tomato salad"
]

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
        "wrong delivery": -5,
        "step penalty": -0.1
    }

    def __init__(self, env_id, grid_dim, task, map_type, n_agent, obs_radius, mode, debug, agent='human'):
        self.env_params = {
            'grid_dim': grid_dim,
            'task': TASKLIST[task],
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

        while True:
            obs=new_obs
            row = [obs['human']]
            self.step += 1
            input_human = input("Input Human: ").strip().split(" ")


            if input_human == ['p']:
                self.save_data(data)
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

            row.append(new_obs['human'])
            row.append(reward['human'])
            row.append(reward['ai'])
            row.append(done['__all__'])

            data.append(copy.deepcopy(row))

            self.env.render()

            if done['__all__']:
                self.save_data(data)
                break

    def save_data(self, data):
        columns = data[0]
        data = data[1:]
        df = pd.DataFrame(data, columns=columns)
        csv_filename = "output.csv"
        df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_dim', type=int, nargs=5, default=[5, 5], help='Grid world size')
    parser.add_argument('--task', type=int, default=6, help='The recipe agent cooks')
    parser.add_argument('--map_type', type=str, default="A", help='The type of map')
    parser.add_argument('--mode', type=str, default="vector", help='The type of observation (vector/image)')
    parser.add_argument('--debug', type=bool, default=True, help='Whether to print debug information and render')

    params = vars(parser.parse_args())

    params['env_id'] = "default_env_id"
    params['n_agent'] = 2
    params['obs_radius'] = 5

    player = Player(**params)
    player.run()