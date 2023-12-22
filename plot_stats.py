import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

random_dir = 'qml-ba/random/results'
dir1 = 'qml-ba/arch1/a2c_5_ppnn/results'
directory = 'paper'

results1 = ['A2C_5_0_05.02.2023_07-00-17.csv',
'A2C_5_1_05.02.2023_11-44-10.csv',
'A2C_5_2_06.02.2023_04-10-44.csv',
'A2C_5_3_06.02.2023_08-47-26.csv',
'A2C_5_4_06.02.2023_13-08-06.csv',
'A2C_5_5_06.02.2023_18-05-59.csv',
'A2C_5_6_07.02.2023_05-37-18.csv',
'A2C_5_7_07.02.2023_12-27-24.csv',
'A2C_5_8_08.02.2023_02-57-47.csv',
'A2C_5_9_08.02.2023_18-51-35.csv']

results2 = ['A2Q_5_0_26.02.2023_07-00-37.csv',
'A2Q_5_1_23.02.2023_10-42-20.csv',
'A2Q_5_2_22.02.2023_02-01-45.csv',
'A2Q_5_3_19.03.2023_07-00-23.csv',
'A2Q_5_4_09.04.2023_07-00-16.csv',
'A2Q_5_5_26.02.2023_07-00-37.csv',
'A2Q_5_6_26.02.2023_07-00-37.csv',
'A2Q_5_7_03.04.2023_11-37-17.csv',
'A2Q_5_8_05.03.2023_07-00-27.csv',
'A2Q_5_9_23.02.2023_02-09-18.csv']

results3 = ['Q2C_5_0_05.03.2023_07-00-28.csv',
'Q2C_5_1_24.02.2023_06-30-34.csv',
'Q2C_5_2_26.02.2023_07-00-42.csv',
'Q2C_5_3_22.02.2023_02-13-33.csv',
'Q2C_5_4_09.04.2023_07-00-21.csv',
'Q2C_5_5_09.04.2023_07-00-21.csv',
'Q2C_5_6_19.03.2023_07-00-20.csv',
'Q2C_5_7_26.02.2023_07-00-42.csv',
'Q2C_5_8_19.03.2023_07-00-20.csv',
'Q2C_5_9_26.02.2023_07-00-42.csv']

results4 = ['Q2Q_5_0_22.02.2023_02-26-13.csv',
'Q2Q_5_1_16.04.2023_07-00-19.csv',
'Q2Q_5_2_16.04.2023_07-00-19.csv',
'Q2Q_5_3_19.03.2023_07-00-33.csv',
'Q2Q_5_4_22.02.2023_18-35-45.csv',
'Q2Q_5_5_22.02.2023_18-35-45.csv',
'Q2Q_5_6_22.02.2023_18-35-44.csv',
'Q2Q_5_7_16.04.2023_07-00-19.csv',
'Q2Q_5_8_09.04.2023_07-00-21.csv',
'Q2Q_5_9_22.02.2023_18-35-47.csv']

random_res = ['random_agent_05.02.2023_07-00-07.csv',
'random_agent_05.02.2023_09-08-02.csv',
'random_agent_05.02.2023_11-36-37.csv',
'random_agent_05.02.2023_14-07-35.csv',
'random_agent_05.02.2023_16-22-37.csv',
'random_agent_05.02.2023_18-29-11.csv',
'random_agent_05.02.2023_20-36-01.csv',
'random_agent_05.02.2023_22-43-12.csv',
'random_agent_06.02.2023_00-50-49.csv',
'random_agent_06.02.2023_02-58-07.csv']

scores1 = []
scores2 = []
scores3 = []
scores4 = []
random_scores = []

# Read the data from each specified file
for file in results1:
    df = pd.read_csv(os.path.join(dir1, file), nrows=26000)
    scores1.append(df["Episode Score"])

for file in results2:
    df = pd.read_csv(os.path.join(directory, file), nrows=26000)
    scores2.append(df["Episode Score"])

for file in results3:
    df = pd.read_csv(os.path.join(directory, file), nrows=26000)
    scores3.append(df["Episode Score"])

for file in results4:
    df = pd.read_csv(os.path.join(directory, file), nrows=26000)
    scores4.append(df["Episode Score"])
#
for file in random_res:
    df = pd.read_csv(os.path.join(random_dir, file), nrows=26000)
    random_scores.append(df["Episode Score"])

# Calculate the average and standard deviation of all episode scores
avg_scores1 = np.mean(scores1, axis=0)
std_scores1 = np.std(scores1, axis=0)

avg_scores2 = np.mean(scores2, axis=0)
std_scores2 = np.std(scores2, axis=0)

avg_scores3 = np.mean(scores3, axis=0)
std_scores3 = np.std(scores3, axis=0)

avg_scores4 = np.mean(scores4, axis=0)
std_scores4 = np.std(scores4, axis=0)

random_avg_scores = np.mean(random_scores, axis=0)
random_std_scores = np.std(random_scores, axis=0)

# Convert numpy array to pandas data frame
avg_scores1 = pd.DataFrame(avg_scores1, columns=["Average Score"])

avg_scores2 = pd.DataFrame(avg_scores2, columns=["Average Score"])

avg_scores3 = pd.DataFrame(avg_scores3, columns=["Average Score"])

avg_scores4 = pd.DataFrame(avg_scores4, columns=["Average Score"])

random_avg_scores = pd.DataFrame(random_avg_scores, columns=["Average Score"])

# Smooth the result
avg_scores1["Average Score"] = avg_scores1["Average Score"].rolling(window=150, min_periods=1).mean()

avg_scores2["Average Score"] = avg_scores2["Average Score"].rolling(window=150, min_periods=1).mean()

avg_scores3["Average Score"] = avg_scores3["Average Score"].rolling(window=150, min_periods=1).mean()

avg_scores4["Average Score"] = avg_scores4["Average Score"].rolling(window=150, min_periods=1).mean()

random_avg_scores["Average Score"] = random_avg_scores["Average Score"].rolling(window=2000, min_periods=1).mean()

# Plot the data
plt.plot(avg_scores1, label="A2C")
plt.fill_between(avg_scores1.index, avg_scores1["Average Score"] - std_scores1, avg_scores1["Average Score"] + std_scores1, alpha=0.2)

plt.plot(avg_scores2, label="HA2Q")
plt.fill_between(avg_scores2.index, avg_scores2["Average Score"] - std_scores2, avg_scores2["Average Score"] + std_scores2, alpha=0.2)

plt.plot(avg_scores3, label="HQ2C")
plt.fill_between(avg_scores3.index, avg_scores3["Average Score"] - std_scores3, avg_scores3["Average Score"] + std_scores3, alpha=0.2)

plt.plot(avg_scores4, label="HQ2Q")
plt.fill_between(avg_scores4.index, avg_scores4["Average Score"] - std_scores4, avg_scores4["Average Score"] + std_scores4, alpha=0.2)

plt.plot(random_avg_scores, label="Random Agent")
plt.fill_between(random_avg_scores.index, random_avg_scores["Average Score"] - random_std_scores, random_avg_scores["Average Score"] + random_std_scores, alpha=0.2)

plt.xlabel("Episode")
plt.ylabel("Average Score")
plt.legend(loc="upper left")
plt.grid(linestyle='-', linewidth=0.5, color='silver')
plt.xlim(0, 26000)
plt.ylim(0, 530)

print(scores1)

# Save plot
plot = "A2C_Hybrid_ZYZ_20k_isitthat"
plt.savefig(plot)

plt.show()
