import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read gradients from the single CSV file
df = pd.read_csv("all_gradient_data.csv")

# Filter the DataFrame for every 1000th episode starting from 1000
episodes_to_plot = range(1000, 8001, 1000)  # Adjusted range
df_filtered = df[df["Episode"].isin(episodes_to_plot)]

# Reshape the filtered DataFrame for plotting
df_melted = df_filtered.melt(id_vars=["Episode"], var_name="Gradient Type", value_name="Gradient")

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(len(episodes_to_plot), rot=-0.45, light=0.7)
g = sns.FacetGrid(
    df_melted,
    row="Episode",
    hue="Episode",
    aspect=15,
    height=0.5,
    palette=pal,
    row_order=sorted(episodes_to_plot, reverse=True),
)

# Draw the densities
g.map(sns.kdeplot, "Gradient", bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "Gradient", clip_on=False, color="w", lw=2, bw_adjust=0.5)

# Refline at y=0
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Function to modify label placement
def label(x, color, label):
    ax = plt.gca()
    ax.text(-0.05, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

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
