import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F


# Quantum Part
nr_qubits = 4
dev = qml.device("default.qubit", wires=nr_qubits)


@qml.qnode(dev)
def qnode(inputs, weights):
    # state encoding
    qml.RX(np.pi * inputs[0] / 4.8, wires=0)
    qml.RX(2 * np.arctan(inputs[1]), wires=1)
    qml.RX(np.pi * inputs[2] / 0.418, wires=2)
    qml.RX(2 * np.arctan(inputs[3]), wires=3)

    # layer
    for j in range(2):
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])

        # rotations
        for i in range(4):
            qml.RZ(weights[i, j, 0], wires=i)
            qml.RY(weights[i, j, 1], wires=i)
            qml.RZ(weights[i, j, 2], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(2)]  # measure


# The Actor (Policy) as a VQC
class QuantumActor(nn.Module):
    def __init__(self, observation_space):
        super(QuantumActor, self).__init__()
        self.nr_qubits = observation_space
        self.nr_layers = 2
        weight_shapes = {"weights": (self.nr_qubits, self.nr_layers, 3)}

        # VQC layer
        self.l1 = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, state):
        expval = self.l1(state)
        prob0 = (expval + 1) / 2  # probability to measure 0
        action_prob = F.softmax(prob0, dim=1)  # activation function Softmax for a probability distribution
        return action_prob  # probability of each action


# The Critic (Value) as a VQC
class QuantumCritic(nn.Module):
    def __init__(self, observation_space):
        super(QuantumCritic, self).__init__()
        self.nr_qubits = observation_space
        self.nr_layers = 2
        weight_shapes = {"weights": (self.nr_qubits, self.nr_layers, 3)}

        # VQC layer
        self.l1 = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, state):
        expval = self.l1(state)
        prob0 = (expval + 1) / 2  # probability to measure 0
        state_value = torch.sum(prob0)
        return state_value  # value of the state


# The A2C Algorithm
class A2C(object):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = 1e-4  # learning rate 0.0001
        self.gamma = 0.99  # discount factor
        self.I = 1
        self.gradient_data = []
        self.new_episode_flag = False

        # init actor
        self.actor = QuantumActor(observation_space)
        # optimize actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        # init critic
        self.critic = QuantumCritic(observation_space)
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

        # Extract gradients for all parameters
        gradients = [p.grad.flatten().cpu().numpy() for p in self.actor.parameters()]
        all_gradients = np.concatenate(gradients)
        if self.new_episode_flag:
            self.gradient_data.append(all_gradients)

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
        max_episode_steps = env._max_episode_steps

        while episodes < max_episodes:
            state = env.reset()
            done = False
            self.I = 1
            episode_steps = 0
            total_score = 0
            self.new_episode_flag = True
            if episodes % 100 == 0:
                print("Episode", episodes)

            while not done:
                episode_steps += 1
                action = self.select_action(state, deterministic=False)
                next_state, reward, done, _ = env.step(action)
                total_score += reward

                if done and episode_steps != max_episode_steps:
                    terminal_state = True
                else:
                    terminal_state = False

                self.learn(state, action, reward, next_state, terminal_state)
                self.new_episode_flag = False
                state = next_state

            # End of an episode
            episodes += 1

            # Append gradients with episode information
            episode_gradients = pd.DataFrame(self.gradient_data)
            episode_gradients["Episode"] = episodes

            # Check if the CSV file exists
            file_exists = os.path.isfile("all_gradient_data.csv")

            # Write to CSV: append if file exists, otherwise write a new file
            episode_gradients.to_csv(
                "all_gradient_data.csv",
                mode='a',  # append data if file exists
                header=not file_exists,  # write header only if file does not exist
                index=False
            )

            self.gradient_data = []  # Clear gradient data for next episode


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env = gym.make(env_name)  # setup environment

    # set random seed for a single seed
    seed = 0
    torch.manual_seed(seed)  # torch
    np.random.seed(seed)  # numpy
    env.seed(seed)  # gym
    env.action_space.seed(seed)  # gym action space

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = A2C(observation_space, action_space)

    agent.train(env, max_episodes=1e5)

    # Read gradients from the single CSV file
    df = pd.read_csv("all_gradient_data.csv")

    # Prepare the DataFrame for plotting
    df = df.melt(id_vars=["Episode"], var_name="Gradient Type", value_name="Gradient")

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(len(df["Episode"].unique()), rot=-0.45, light=0.7)
    g = sns.FacetGrid(
        df,
        row="Episode",
        hue="Episode",
        aspect=15,
        height=0.5,
        palette=pal,
        row_order=df["Episode"].unique()[::-1],
    )

    # Draw the densities in a few steps
    g.map(
        sns.kdeplot,
        "Gradient",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    g.map(sns.kdeplot, "Gradient", clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # Refline at y=0
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Modify the label function for episode labels
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            -0.05,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )


    g.map(label, "Gradient")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(xlabel="Gradient")
    g.despine(bottom=True, left=True)
    g.fig.text(-0.1, 0.5, 'Episodes', va='center', rotation='vertical', fontsize=12)

    plt.show()
