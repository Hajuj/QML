import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# random_dir = 'qml-ba/random/results'
# dir1 = 'qml-ba/arch1/a2c_4/results'
# directory = 'qml-ba/arch3/a2c_4/results'
#
# results1 = ['A2C_4_0_05.02.2023_07-00-17.csv',
# 'A2C_4_1_05.02.2023_11-37-31.csv',
# 'A2C_4_2_06.02.2023_01-59-54.csv',
# 'A2C_4_3_06.02.2023_06-19-01.csv',
# 'A2C_4_4_06.02.2023_12-26-45.csv',
# 'A2C_4_5_06.02.2023_23-15-23.csv',
# 'A2C_4_6_07.02.2023_05-27-40.csv',
# 'A2C_4_7_07.02.2023_09-47-11.csv',
# 'A2C_4_8_08.02.2023_00-17-15.csv',
# 'A2C_4_9_08.02.2023_04-33-32.csv']
#
# results2 = ['A2Q_4_0_22.02.2023_20-42-58.csv',
# 'A2Q_4_1_22.02.2023_20-44-00.csv',
# 'A2Q_4_2_22.02.2023_20-44-45.csv',
# 'A2Q_4_3_22.02.2023_20-45-08.csv',
# 'A2Q_4_4_22.02.2023_20-45-28.csv',
# 'A2Q_4_5_22.02.2023_20-46-42.csv',
# 'A2Q_4_6_22.02.2023_20-47-10.csv',
# 'A2Q_4_7_22.02.2023_20-47-30.csv',
# 'A2Q_4_8_22.02.2023_20-48-45.csv',
# 'A2Q_4_9_22.02.2023_20-49-08.csv']
#
# results3 = ['Q2C_4_0_23.02.2023_18-29-07.csv',
# 'Q2C_4_1_23.02.2023_18-59-54.csv',
# 'Q2C_4_2_23.02.2023_19-00-33.csv',
# 'Q2C_4_3_23.02.2023_19-04-18.csv',
# 'Q2C_4_4_23.02.2023_19-04-58.csv',
# 'Q2C_4_5_23.02.2023_19-56-15.csv',
# 'Q2C_4_6_23.02.2023_19-57-15.csv',
# 'Q2C_4_7_23.02.2023_20-31-54.csv',
# 'Q2C_4_8_23.02.2023_20-32-40.csv',
# 'Q2C_4_9_23.02.2023_20-33-17.csv']
#
# results4 = ['Q2Q_4_0_25.02.2023_15-43-18.csv',
# 'Q2Q_4_1_25.02.2023_15-43-56.csv',
# 'Q2Q_4_2_25.02.2023_15-44-25.csv',
# 'Q2Q_4_3_25.02.2023_15-44-50.csv',
# 'Q2Q_4_4_25.02.2023_15-45-11.csv',
# 'Q2Q_4_5_25.02.2023_15-45-44.csv',
# 'Q2Q_4_6_25.02.2023_15-46-20.csv',
# 'Q2Q_4_7_25.02.2023_15-46-38.csv',
# 'Q2Q_4_8_25.02.2023_15-47-01.csv',
# 'Q2Q_4_9_25.02.2023_15-47-23.csv']
#
# random_res = ['random_agent_05.02.2023_07-00-07.csv',
#               'random_agent_05.02.2023_09-08-02.csv',
#               'random_agent_05.02.2023_11-36-37.csv',
#               'random_agent_05.02.2023_14-07-35.csv',
#               'random_agent_05.02.2023_16-22-37.csv',
#               'random_agent_05.02.2023_18-29-11.csv',
#               'random_agent_05.02.2023_20-36-01.csv',
#               'random_agent_05.02.2023_22-43-12.csv',
#               'random_agent_06.02.2023_00-50-49.csv',
#               'random_agent_06.02.2023_02-58-07.csv']

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


def read_and_process_scores(file_list, directory, window_size):
    scores = []
    total_steps = []
    for file in file_list:
        df = pd.read_csv(os.path.join(directory, file), nrows=26000)
        scores.append(df["Episode Score"])
        total_steps.append(df["Episode Score"].cumsum())  # Cumulative sum for total steps

    avg_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)

    avg_total_steps = np.mean(total_steps, axis=0)  # Average of total steps

    # Convert numpy array to pandas data frame and smooth
    avg_scores_df = pd.DataFrame(avg_scores, columns=["Average Score"])
    avg_scores_df["Average Score"] = avg_scores_df["Average Score"].rolling(window=window_size, min_periods=1).mean()

    return avg_scores_df, std_scores, avg_total_steps


# Processing each agent's results
avg_scores1, std_scores1, total_steps1 = read_and_process_scores(results1, dir1, 150)
avg_scores2, std_scores2, total_steps2 = read_and_process_scores(results2, directory, 150)
avg_scores3, std_scores3, total_steps3 = read_and_process_scores(results3, directory, 150)
avg_scores4, std_scores4, total_steps4 = read_and_process_scores(results4, directory, 150)
random_avg_scores, random_std_scores, random_total_steps = read_and_process_scores(random_res, random_dir, 2000)


# Extend the random agent's data to 1 million steps if needed
max_steps = 1_000_000
if random_total_steps[-1] < max_steps:
    last_known_score = random_avg_scores.iloc[-1]["Average Score"]
    last_known_std = random_std_scores[-1]

    # Ensure that additional_steps is an integer
    additional_steps = int(max_steps - random_total_steps[-1]) + 1

    # Generate extended data
    extended_scores = np.full(additional_steps, last_known_score)
    extended_std = np.full(additional_steps, last_known_std)

    # Extend the average scores dataframe
    extended_scores_df = pd.DataFrame({'Average Score': extended_scores})
    random_avg_scores = pd.concat([random_avg_scores, extended_scores_df]).reset_index(drop=True)

    # Extend the standard deviation array
    random_std_scores = np.concatenate((random_std_scores, extended_std))

    # Update the total_steps array
    extended_steps_array = np.arange(random_total_steps[-1] + 1, max_steps + 1)
    random_total_steps = np.concatenate((random_total_steps, extended_steps_array))


# Plotting
plt.figure(figsize=(10, 6))


def plot_data(x, y, std, label):
    plt.plot(x, y["Average Score"], label=label, linewidth=2.5)
    plt.fill_between(x, y["Average Score"] - std, y["Average Score"] + std, alpha=0.2)


plot_data(total_steps1, avg_scores1, std_scores1, "A2C")
plot_data(total_steps2, avg_scores2, std_scores2, "HA2Q")
plot_data(total_steps3, avg_scores3, std_scores3, "HQ2C")
plot_data(total_steps4, avg_scores4, std_scores4, "HQ2Q")
plot_data(random_total_steps, random_avg_scores, random_std_scores, "Random Agent")

plt.xlabel("Total Steps", fontsize=17)
plt.ylabel("Average Score", fontsize=17)
plt.legend(loc="upper left")
plt.grid(linestyle='-', linewidth=0.5, color='silver')
plt.xlim(0, max_steps)  # Set upper limit to 1 million
plt.ylim(0, 530)


# Save plot
plot = "plot_hybrid_1m"
plt.savefig(plot)

plt.show()
