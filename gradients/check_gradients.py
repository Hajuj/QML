import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        action_prob = F.softmax(prob0, dim=0)  # activation function Softmax for a probability distribution
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


# def plot_gradients(model, states):
#     gradients = []
#
#     for state in states:
#         state = torch.tensor(state, dtype=torch.float32)
#         state.requires_grad = False
#         output = model(state)
#         output = torch.sum(output)  # Sum the output if it is a vector
#         output.backward()  # Compute the gradient
#         # weights = model.state_dict().get('l1.weights')
#         weights = model.l1
#         print("Weights: ", weights)
#         grad = weights.grad  # Get the gradient
#         print("Grad: ", grad)
#         gradients.append(grad.numpy())
#
#     # Flatten the list of gradients for plotting
#     gradients = np.concatenate(gradients).ravel()
#
#     # Plot the gradients
#     plt.hist(gradients, bins=50, density=True, alpha=0.7, color='g')
#     plt.xlabel('Gradient values')
#     plt.ylabel('Density')
#     plt.title('Gradients of the Quantum Circuit')
#     plt.grid(True)
#     plt.show()
#
#
# # Create random states as inputs
# np.random.seed(0)  # for reproducibility
# random_states = np.random.rand(100, nr_qubits)  # 100 random states
#
# # Initialize the QuantumActor model and plot the gradients
# actor = QuantumActor(nr_qubits)
# plot_gradients(actor, random_states)

############################################################

# Generate a range of random input states
# num_samples = 200
# input_states = np.random.rand(num_samples, nr_qubits)
#
# # Generate a range of random weights
# num_weights = 100
# weights = np.random.rand(num_weights, nr_qubits, 2, 3) * 2 * np.pi
#
# # Compute gradients for each weight
# gradients = np.zeros((num_weights, nr_qubits, 2, 3))
# for i, weight in enumerate(weights):
#     for j, state in enumerate(input_states):
#         weight_torch = np.array(weight, requires_grad=True)
#         state_torch = torch.tensor(state, requires_grad=False, dtype=torch.float32)
#         q_values = qnode(state_torch, weight_torch.detach().numpy())
#         q_values = torch.sum(q_values)  # Sum if the output is a vector
#         q_values.backward()  # Compute the gradient
#         gradients[i] = weight_torch.grad.detach().numpy()
#
# # Compute the mean and standard deviation of gradients
# mean_gradients = np.mean(np.abs(gradients), axis=0)
# std_gradients = np.std(gradients, axis=0)
#
# # Plot mean gradient for each parameter
# plt.figure(figsize=(10, 5))
# for i in range(nr_qubits):
#     for j in range(2):
#         for k in range(3):
#             plt.plot(mean_gradients[i, j, k], label=f'Qubit {i}, Layer {j}, Param {k}')
#
# plt.yscale('log')
# plt.xlabel('Parameter index')
# plt.ylabel('Mean gradient magnitude')
# plt.title('Mean Gradient Magnitude for Random Weights')
# plt.legend()
# plt.grid(True)
# plt.show()

##############################################################

# Initialize the parameters
# nr_layers = 2
# shape = (nr_qubits, nr_layers, 3)
#
# # Compute the gradient
# gradient = qml.grad(qnode, argnum=0)
#
#
# def gradient_norm(params):
#     grad_vals = gradient(params)
#     norm = np.linalg.norm(grad_vals)
#     return norm
#
#
# # Evaluate gradient norms over multiple random initializations
# gradient_norms = []
# num_random_trials = 100
#
# for _ in range(num_random_trials):
#     random_weights = np.random.randn(*shape)
#     inputs = np.random.random(size=4)
#     gradient_norms.append(gradient_norm([inputs, random_weights]))
#
# # Plot the gradient norms
# plt.figure(figsize=(10, 6))
# plt.plot(gradient_norms, 'o-', markersize=5)
# plt.title('Gradient Norms over Multiple Random Initializations')
# plt.xlabel('Run #')
# plt.ylabel('Gradient Norm')
# plt.grid(True)
# plt.show()

##############################################################

np.random.seed(42)

# Compute gradient
gradient = qml.grad(qnode, argnum=1)  # Gradients with respect to the weights

# Generate random inputs and weights
num_samples = 50
inputs = 2 * np.pi * np.random.rand(num_samples, nr_qubits)  # Random input values in [0, 2*pi]
weights_shape = (nr_qubits, 2, 3)
weights = 2 * np.pi * np.random.rand(*weights_shape)  # Random weight values in [0, 2*pi]

# Number of steps and learning rate for gradient descent
num_steps = 2
lr = 1e-4

# Compute jacobian for the updated weights
jacobian = qml.jacobian(qnode, argnum=1)

gradient_values_over_steps = []

for step in range(num_steps):
    total_gradient = np.zeros_like(weights)
    for i in range(num_samples):
        jacobian_matrix = jacobian(inputs[i], weights)
        gradient_magnitudes = np.linalg.norm(jacobian_matrix, axis=0)  # Take the norm over output dimensions for each parameter
        total_gradient += gradient_magnitudes
        gradient_values_over_steps.extend(gradient_magnitudes.flatten())

    # Update weights using gradient descent
    weights -= lr * total_gradient / num_samples  # Average gradient over samples

# Plotting
plt.hist(gradient_values_over_steps, bins=30)
plt.title("Gradient Values Over Steps")
plt.xlabel("Gradient Value")
plt.ylabel("Frequency")
plt.show()
