import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pickle
import sys
sys.path.append("../..")
import recipe_planner


recipe = "salad"
map_ = "open-divider"
models = [
    "_model1-bd_model2-bd",
    "_model1-up_model2-up",
    "_model1-fb_model2-fb",
    "_model1-dc_model2-dc",
    "_model1-greedy_model2-greedy"
    ]
model_key = {
    "_model1-bd_model2-bd": "BD (ours)",
    "_model1-up_model2-up": "UP",
    "_model1-fb_model2-fb": "FB",
    "_model1-dc_model2-dc": "D&C",
    "_model1-greedy_model2-greedy": "Greedy",
}

seed = 1
num_agents = 2
path_pickles = os.path.join(os.getcwd(), 'pickles')  # Ensure your files are in a "pickles" folder
path_save = os.path.join(os.getcwd(), f'graphs_agents{num_agents}')
os.makedirs(path_save, exist_ok=True)


# Helper functions
def get_time_steps(data, recipe):
    try:
        return data['num_completed_subtasks'].index(5) + 1
    except:
        return 100


def get_completion(data, recipe, t):
    completion = data['num_completed_subtasks']
    try:
        end_indx = completion.index(5) + 1
        completion = completion[:end_indx]
    except:
        pass
    if len(completion) < 100:
        completion += [data['num_completed_subtasks_end']] * (100 - len(completion))
    return completion[t] / 5


# Initialize data containers
time_steps_data = []
completion_data = {model_key[model]: [] for model in models}

# Process each model
for model in models:
    file_name = f"{map_}_{recipe}_agents{num_agents}_seed{seed}{model}.pkl"
    file_path = os.path.join(path_pickles, file_name)

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded data from {file_name}")

        # Get time steps
        time_steps = get_time_steps(data, recipe)
        time_steps_data.append((model_key[model], time_steps))

        # Get completion over time
        completion_data[model_key[model]] = [get_completion(data, recipe, t) for t in range(100)]
    else:
        print(f"File not found: {file_name}")

# Plot Time Steps Graph with Legend
sns.set_style('ticks')
sns.set_context('talk', font_scale=1)

plt.figure(figsize=(7, 5))
time_steps_df = pd.DataFrame(time_steps_data, columns=["Model", "Time Steps"])
sns.barplot(x="Model", y="Time Steps", data=time_steps_df, palette="deep").set(
    ylabel="Time", ylim=[0, 100]
)
plt.legend(title="Model")
sns.despine()
plt.tight_layout()
time_steps_path = os.path.join(path_save, f"time_steps_{recipe}_{map_}_all_models.png")
plt.savefig(time_steps_path)
plt.close()
print(f"Saved time steps graph with legend at: {time_steps_path}")

# Plot Completion Graph with Legend
plt.figure(figsize=(10, 8))
for model_label, comp_data in completion_data.items():
    sns.lineplot(x=range(100), y=comp_data, label=model_label, linewidth=2)
plt.xlabel('Steps')
plt.ylabel('Completion')
plt.ylim([0, 1])
plt.legend(title="Model")
plt.tight_layout()
completion_path = os.path.join(path_save, f"completion_{recipe}_{map_}_all_models.png")
plt.savefig(completion_path)
plt.close()
print(f"Saved completion graph with legend at: {completion_path}")

