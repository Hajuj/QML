import csv
import datetime
import os

import gym
import numpy as np


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self):
        # random choice
        action = np.random.choice(range(self.action_space), p=[0.5, 0.5])
        return action

    def train(self, env, max_episodes):
        episodes = 0
        stats = []

        while episodes < max_episodes:
            env.reset()
            done = False
            self.I = 1
            episode_steps = 0
            total_score = 0

            # run the episode
            while not done:
                episode_steps += 1
                action = self.select_action()
                _, reward, done, _ = env.step(action)
                total_score += reward

            episodes += 1

            # add stats
            stats.append([episodes, total_score, env_name, max_episodes, seed])

            # save stats to CSV file
            with open(os.path.join(directory, file_name), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['Episode', 'Episode Score', 'Environment', 'Max Episodes', 'Seed'])
                writer.writerows(stats)


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env = gym.make(env_name)  # setup environment

    # set random seed
    for seed in range(10):
        np.random.seed(seed)  # numpy
        env.seed(seed)  # gym
        env.action_space.seed(seed)  # gym action space

        action_space = env.action_space.n

        # create directory if it doesn't exist
        directory = 'results'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # create the file name with time stamp
        file_name = "random_agent.csv".split(".")[0] + '_' + datetime.datetime.now().strftime("%d.%m.%Y_%H-%M-%S") + ".csv"

        agent = RandomAgent(action_space)

        agent.train(env, max_episodes=1e5)
