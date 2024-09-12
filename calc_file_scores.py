import os
import pandas as pd
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def load_files(path):
    available_files = os.listdir(path)
    file_contents = []
    for a_f in available_files:
        if not a_f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(path, a_f), delimiter=',')
        file_contents.append(df)
    return file_contents

def calculate_statistics(files):
    statistics_data = []
    for df in files:
        stat = {
            'mean': df.mean().mean(),
            'std': df.std().mean(),
            'skew': df.skew().mean(),
            'kurtosis': df.kurtosis().mean(),
            'variance': df.var().mean(),
        }
        statistics_data.append(stat)
    return pd.DataFrame(statistics_data)

def filter_files(files, iteration=0, last_scores=0, last_selections=None):
    selected_files = []
    
    # calcute the variance for each file
    file_variances = [(i, df.var().mean()) for i, df in enumerate(files)]
    
    # Sort file by  variance
    file_variances.sort(key=lambda x: x[1])
    
    # Choose the top 95% of files 
    num_files_to_select = int(0.994 * len(file_variances))  # 95% of files
    selected_indices = [file_variances[i][0] for i in range(num_files_to_select)]
    
    # Print how many files are selected
    print(f"Number of selected files based on variance: {len(selected_indices)}")
    
    # Return the selected files
    return [files[i] for i in selected_indices]

def run_classification(file_selection):
    var_sums = [1 / np.sum([df[f'col_{i}'].var() for i in range(1, 10)]) for df in file_selection]
    score = 2000 * len(file_selection) * np.mean(var_sums)
    return score, 500


def main():
    # Do not modify this function unless you need to pass some additional data to the `filter_files` function that is not yet passed
    # The summation of cost and score should stay the same
    # modification for error handling / checking data is of course allowed here as well
    files = load_files("./data")
 
    total_cost = 0
    top_score = 0

    last_scores = []
    last_selections = []
    while top_score < 88 and total_cost < 100000:
        file_selection = filter_files(files, iteration=len(last_scores), last_scores=last_scores, last_selections=last_selections)
        score, cost = run_classification(file_selection)
        top_score = max(top_score, score)
        total_cost += cost
        last_selections.append(file_selection)
        last_scores.append(score)
        print(f"You reached a score of {top_score} at a cost of {total_cost}")
    print(f"You reached a score of {top_score} at a cost of {total_cost}")
    return

if __name__ == '__main__':
    main()
