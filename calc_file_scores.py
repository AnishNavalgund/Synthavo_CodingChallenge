import os
import pandas as pd
import numpy as np

# Load CSV files
def load_files(path):
    available_files = os.listdir(path)
    file_contents = []
    for a_f in available_files:
        if not a_f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(path, a_f), delimiter=',')
        file_contents.append(df)
    return file_contents

# Calculate variance for each file
def calculate_variance(files):
    variances = []
    for df in files:
        variance = df.var().mean()  # Average variance across columns
        variances.append(variance)
    return np.array(variances)


def filter_files(files, iteration=0, last_scores=0, last_selections=None):
    variances = calculate_variance(files)
    
    sorted_indices = np.argsort(variances)
    num_files_to_select = int(len(files) * 0.9)  
    selected_files = [files[i] for i in sorted_indices[:num_files_to_select]]
    
    # Print selected files and their variance values
    print(f"Number of selected files: {len(selected_files)}")
    return selected_files

# Run classification 
def run_classification(file_selection):
    var_sums = [1 / np.sum([df[f'col_{i}'].var() for i in range(1, 10)]) for df in file_selection]
    score = 2000 * len(file_selection) * np.mean(var_sums)
    return score, 500

# Main function
def main():
    files = load_files("./data")
    total_cost = 0
    top_score = 0

    last_scores = []
    last_selections = []
    iteration = 0
    
    while top_score < 88 and total_cost < 100000:
        file_selection = filter_files(files, iteration, last_scores=last_scores, last_selections=last_selections)
        score, cost = run_classification(file_selection)
        top_score = max(top_score, score)
        total_cost += cost
        last_selections.append(file_selection)
        last_scores.append(score)
        
        iteration += 1  # Track iterations
        
        print(f"You reached a score of {top_score} at a cost of {total_cost}")
    
    print(f"Final score: {top_score} at a total cost of {total_cost}")

if __name__ == '__main__':
    main()
