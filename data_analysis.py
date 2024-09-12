import os
import pandas as pd
import numpy as np

# data path
DATA_DIR = './data'

# Load files
def load_files(path):

    file_contents = [] # list to store files
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')] # select all cvs files
    
    for file in csv_files:
            df = pd.read_csv(os.path.join(path, file))
            file_contents.append((file, df))  

    print(f"Loaded {len(file_contents)} files")
    return file_contents

# statistics calulations
def calculate_statistics_for_files(files):
    statistics_data = []
    for file_name, df in files:
        stats = {}
        stats['file_name'] = file_name
        stats['mean'] = df.mean().mean()  # mean of all columns
        stats['std'] = df.std().mean()  # average standard deviation of all columns
        stats['skew'] = df.skew().mean()  # average skewness of all columns
        stats['kurtosis'] = df.kurtosis().mean()  # average kurtosis of all columns
        stats['variance'] = df.var().mean()  # average variance of all columns
        statistics_data.append(stats)
    
    print(f"Calculated statistics for {len(statistics_data)} files.")
    return pd.DataFrame(statistics_data)

# analzyew data
def analyze_data():

    files = load_files(DATA_DIR) # load in files
    stats_df = calculate_statistics_for_files(files) # calculate statistics
    print(stats_df) # print statistics
    
if __name__ == "__main__":
    analyze_data()
