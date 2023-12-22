import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import seaborn as sns
import pandas as pd

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


np.random.seed(42)

# Generate random inputs and weights
num_samples = 10
inputs = 2 * np.pi * np.random.rand(num_samples, nr_qubits)  # Random input values in [0, 2*pi]
weights_shape = (nr_qubits, 2, 3)
weights = 2 * np.pi * np.random.rand(*weights_shape)  # Random weight values in [0, 2*pi]

# Number of steps and learning rate for gradient descent
num_steps = 100
lr = 0.01

# Compute jacobian for the updated weights
jacobian = qml.jacobian(qnode, argnum=1)

gradient_values_over_steps = []

for step in range(num_steps):
    total_gradient = np.zeros_like(weights)
    for i in range(num_samples):
        jacobian_matrix = jacobian(inputs[i], weights)
        gradient_magnitudes = np.linalg.norm(jacobian_matrix,
                                             axis=0)  # Take the norm over output dimensions for each parameter
        total_gradient += gradient_magnitudes
    print(total_gradient)

    if step % 10 == 0:
        gradient_values_over_steps.append(total_gradient.flatten())

    # Update weights using gradient descent
    weights -= lr * total_gradient / num_samples  # Average gradient over samples

# print(gradient_values_over_steps)
# # Plotting
# plt.hist(gradient_values_over_steps, bins=30)
# plt.title("Gradient Values Over Steps")
# plt.xlabel("Gradient Value")
# plt.ylabel("Frequency")
# plt.show()

# Convert gradients to DataFrame
df = pd.DataFrame(gradient_values_over_steps).T.melt(var_name="epoch", value_name="gradient")

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(len(gradient_values_over_steps), rot=-0.45, light=0.7)
g = sns.FacetGrid(
    df,
    row="epoch",
    hue="epoch",
    aspect=15,
    height=0.5,
    palette=pal,
    row_order=df["epoch"].unique()[::-1],
)

# Draw the densities in a few steps
g.map(
    sns.kdeplot,
    "gradient",
    bw_adjust=0.5,
    clip_on=False,
    fill=True,
    alpha=1,
    linewidth=1.5,
)
g.map(sns.kdeplot, "gradient", clip_on=False, color="w", lw=2, bw_adjust=0.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(
        0,
        0.2,
        label,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


g.map(label, "gradient")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-0.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
# g.tight_layout()
plt.show()
