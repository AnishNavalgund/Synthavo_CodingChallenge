import os
import pandas as pd
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform


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

# calculate pairwise Euclideandistances 
def calculate_distance_matrix(stats_df):
    stats_array = stats_df[['mean', 'std', 'skew', 'kurtosis', 'variance']].values
    return squareform(pdist(stats_array, metric='euclidean'))

def filter_files(files, distance_matrix):
    selected_indices = set()
    
    # Get the indices that sort the flattened distance matrix
    sorted_indices = np.argsort(distance_matrix.ravel())
    
    #  small distabce files 
    for flat_idx in sorted_indices[:9*len(files)//10]:
        i, j = np.unravel_index(flat_idx, distance_matrix.shape)
        selected_indices.add(i)
        selected_indices.add(j)
    
    #  large distance files
    for flat_idx in sorted_indices[-4*len(files)//10:]:
        i, j = np.unravel_index(flat_idx, distance_matrix.shape)
        selected_indices.add(i)
        selected_indices.add(j)
    
    # Print how many files are selected
    print(f"Number of selected files: {len(selected_indices)}")
    
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
    stats_df = calculate_statistics(files)  
    distance_matrix = calculate_distance_matrix(stats_df)  
    total_cost = 0
    top_score = 0

    last_scores = []
    last_selections = []
    while top_score < 88 and total_cost < 100000:
        file_selection = filter_files(files, distance_matrix) # filter_files(files, last_scores=last_scores, last_selections=last_selections)
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
