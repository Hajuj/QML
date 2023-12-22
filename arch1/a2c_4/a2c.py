import csv
import datetime
import os

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# The Actor (Policy) as a Neural Network
class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(observation_space, hidden_layers)  # input layer (observations), hidden layers
        self.l2 = nn.Linear(hidden_layers, action_space)  # hidden layers, output layer (actions)

    def forward(self, state):
        state = F.relu(self.l1(state))  # activation function RelU
        action_prob = F.softmax(self.l2(state), dim=1)  # activation function Softmax for a probability distribution
        return action_prob  # probability of each action


# The Critic (Value) as a Neural Network
class Critic(nn.Module):
    def __init__(self, observation_space, hidden_layers):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(observation_space, hidden_layers)  # input layer, hidden layers
        self.l2 = nn.Linear(hidden_layers, 1)  # hidden layers, output layer

    def forward(self, state):
        state = F.relu(self.l1(state))  # activation function RelU
        state_value = self.l2(state)  # get the state value
        return state_value  # value of the state


# The A2C Algorithm
class A2C(object):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_layers = 4  # number of neurons in hidden layer
        self.lr = 1e-4  # learning rate 0.0001
        self.gamma = 0.99  # discount factor
        self.I = 1

        # init actor
        self.actor = Actor(observation_space, action_space, self.hidden_layers)
        # optimize actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        # init critic
        self.critic = Critic(observation_space, self.hidden_layers)
        # optimize critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def select_action(self, state, deterministic):
        state = torch.from_numpy(state).float().unsqueeze(0)  # convert state to float tensor and add one dimension
        action_prob = self.actor(state).detach().numpy().flatten()  # let the actor predict action probabilities

        if deterministic:  # deterministic policy
            action = np.argmax(action_prob)  # select action with the highest probability
            return action
        else:  # stochastic policy
            action = np.random.choice(range(self.action_space), p=action_prob)  # action from probability distribution
            return action

    def learn(self, state, action, reward, next_state, terminal_state):
        state = torch.from_numpy(state).float().unsqueeze(0)  # convert state to float tensor and add one dimension
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)  # convert next state also
        value_function = self.critic(state).flatten()  # critic state value of current state (flatten() removes one dim)
        next_value_function = self.critic(next_state).flatten()  # critic state value of next state
        log_prob = torch.log(self.actor(state).flatten()[action])  # log pi(s, a)

        with torch.no_grad():  # TD-Target without gradient
            td_target = reward + self.gamma * (1 - terminal_state) * next_value_function  # TD-Target

        # update actor
        advantage = td_target - value_function  # advantage function in A2C (TD-Error)
        actor_loss = -self.I * advantage.detach() * log_prob  # calculate actor loss

        # backpropagation actor
        self.actor_optimizer.zero_grad()  # clears old gradients from the last step
        actor_loss.backward()  # computes the derivative of the loss w.r.t. the parameters
        self.actor_optimizer.step()  # causes the optimizer to take a step based on the gradients of the parameters

        # update critic
        critic_loss = advantage ** 2  # calculate critic loss (MSE of the advantage)

        # backpropagation critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.I *= self.gamma  # represent the gamma^t in th policy gradient theorem

    def train(self, env, max_episodes):
        episodes = 0
        stats = []
        max_episode_steps = env._max_episode_steps

        while episodes < max_episodes:
            state = env.reset()
            done = False
            self.I = 1
            episode_steps = 0
            total_score = 0

            # run the episode
            while not done:
                episode_steps += 1
                action = self.select_action(state, deterministic=False)
                next_state, reward, done, _ = env.step(action)
                total_score += reward

                if done and episode_steps != max_episode_steps:
                    terminal_state = True
                else:
                    terminal_state = False

                # learn every step -> Temporal-Difference Learning
                self.learn(state, action, reward, next_state, terminal_state)

                state = next_state

            episodes += 1

            # add stats
            stats.append([episodes, total_score, env_name, max_episodes, self.lr, self.gamma,
                          self.hidden_layers, seed])

            # save stats to CSV file
            with open(os.path.join(directory, file_name), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['Episode', 'Episode Score', 'Environment', 'Max Episodes', 'Learning Rate', 'Gamma',
                     'Hidden Layers', 'Seed'])
                writer.writerows(stats)


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env = gym.make(env_name)  # setup environment

    # set random seed
    for seed in range(10):
        torch.manual_seed(seed)  # torch
        np.random.seed(seed)  # numpy
        env.seed(seed)  # gym
        env.action_space.seed(seed)  # gym action space

        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        # create directory if it doesn't exist
        directory = 'results'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # create the file name with time stamp
        file_name = "A2C_4.csv".split(".")[0] + "_{}".format(seed) + '_' + datetime.datetime.now().strftime("%d.%m.%Y_%H-%M-%S") + ".csv"

        agent = A2C(observation_space, action_space)

        agent.train(env, max_episodes=1e5)
