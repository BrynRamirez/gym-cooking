import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import pandas as pd
from itertools import product

import numpy as np
import pickle
import sys
from collections import defaultdict
import os
sys.path.append("../..")
import recipe_planner
import seaborn as sns

# NOTE: THIS SCRIPT IS ONLY WRITTEN FOR 2 AGENTS


# Define parameters
models = ["bd", "up", "dc", "fb", "greedy"]
level = "open-divider_salad"
num_agents = 2
seed = 1
path_pickles = os.path.join(os.getcwd(), 'pickles')
path_save = os.path.join(os.getcwd(), 'heatmaps')

if not os.path.exists(path_save):
    os.mkdir(path_save)

# Initialize dataframe
df = []

for model1, model2 in product(models, repeat=2):
    # Construct filename based on the required pattern
    fname = f"{level}_agents{num_agents}_seed{seed}_model1-{model1}_model2-{model2}.pkl"
    full_path = os.path.join(path_pickles, fname)

    # Check if the file exists
    if os.path.exists(full_path):
        try:
            # Load the data
            data = pickle.load(open(full_path, "rb"))
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        # Collect relevant information
        info = {
            'model1': model1,
            'model2': model2,
            'was_successful': data.get('was_successful', False),
            'time_steps': len(data.get('num_completed_subtasks', [])) if data.get('was_successful') else 100
        }
        df.append(info)
    else:
        print(f"File not found: {fname}")

# Convert to DataFrame
df = pd.DataFrame(df)

# Generate heatmap or analysis
if not df.empty:
    ax = sns.heatmap(
        df.pivot('model1', 'model2', 'time_steps'),
        cmap="Blues",
        annot=True,
        fmt=".1f"
    )
    ax.set_title('Time Steps Heatmap')
    plt.savefig(os.path.join(path_save, "heatmap_filtered.pdf"))
    plt.show()
else:
    print("No matching files found. Heatmap cannot be generated.")
