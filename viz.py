import argparse
import pandas as pd
from tqdm import tqdm
import prolif as plf
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_types
from pathlib import Path
import nbimporter
import extraction
from extraction import *
from extraction import process_multiple_pdb_files
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import json

def build_smooth_distribution(dataframe, merge_distributions, group_ids=None, columns=None, smooth_distribution=0, 
                              smooth_tolerance=3, bin_size=0.1, n_bins=50, range=None):
    if group_ids is not None:  
        dataframe = dataframe[dataframe['Residue'].isin(group_ids)] # if no group_ids is specified, we calculate distributions across all residues. If specidied, we filter dataframe for only those residues
    
    if columns is None: # Default case of all columns
        columns = [col for col in dataframe.columns if col != 'Residue']
    
    distributions = {} #Dictionary to store distributions
    
    for column in columns: # Either default case or list of columns we specify
        data = dataframe[column].dropna()
        
        # If the range is not specified, calculate it from the data
        if range is None:
            data_range = (data.min(), data.max())
        else:
            data_range = range
            
        bins = np.linspace(data_range[0], data_range[1], n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Initialize the distribution
        dist = np.zeros(n_bins)
        
        for value in data:
            if smooth_distribution > 0:
                gaussian_contributions = norm.pdf(bin_centers, loc=value, scale=smooth_distribution)
                gaussian_contributions[gaussian_contributions < norm.pdf(smooth_tolerance)] = 0
                dist += gaussian_contributions
            else:
                # Place the value in the appropriate bin without smoothing
                bin_index = np.digitize(value, bins) - 1
                if 0 <= bin_index < len(dist):
                    dist[bin_index] += 1
        dist /= dist.sum() * bin_size  # Normalize the distribution
        
        if merge_distributions: # If we want to merge across columns and multiple columns are specified
            if 'Merged' not in distributions: # For 1st merge
                distributions['Merged'] = dist
            else:
                distributions['Merged'] += dist # Adding next column's distribution
        else: # If we want the distributions separately for each column, output as a dictionary (append to it) with entries for each column
            distributions[column] = dist
    
    return distributions, bin_centers


def process_dataframes(folder, box_size, ligand_name, ligand_path, columns, smooth_distribution, smooth_tolerance, bin_size, n_bins, merge_distributions, group_ids, range, sliding_window):
    file_list = [f for f in os.listdir(folder) if f.endswith('.pdb')]
    all_distributions = []
    all_bin_centers = []

    pdb_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pdb')]
    df  = process_multiple_pdb_files(pdb_files, box_size, ligand_name, ligand_path)
    print("Processing done")

    # for file in file_list:
    #     filepath = os.path.join(folder, file)
        
        
    distributions, bin_centers = build_smooth_distribution(df, merge_distributions = merge_distributions, group_ids = group_ids, columns = columns, smooth_distribution = smooth_distribution, smooth_tolerance = smooth_tolerance, bin_size = bin_size, n_bins = n_bins, range=None)
    if merge_distributions:
            # Append the merged distribution to the list
            all_distributions.append(distributions['Merged'])
            # Combine all merged distributions into a single plot
            merged_distribution = np.mean(all_distributions, axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(bin_centers, merged_distribution, label='Merged Distribution', color='blue')
            plt.fill_between(bin_centers, merged_distribution, color='blue', alpha=0.2)
            plt.title('Merged Distribution Across All Dataframes')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.show()
            #plt.savefig('output_plot.png')
            #print("SAVED")


    else:
        # If not merging, calculate the mean and variance over the distributions
        for col in distributions:
                all_distributions.append(distributions[col])
                all_bin_centers.append(bin_centers)

        # mean_distribution = np.mean(all_distributions, axis=0)
        # variance_distribution = np.var(all_distributions, axis=0)
        
        # Plotting the results
        for i, col in enumerate(distributions.keys()):
            plt.figure(figsize=(10, 6))
            
            plt.plot(all_bin_centers[i], all_distributions[i], label=f'Distribution of {col}', color='blue')
            if sliding_window:
                smoothed_distribution = np.convolve(distributions[col], np.ones(sliding_window)/sliding_window, mode='same')
                plt.plot(all_bin_centers[i], smoothed_distribution, label=f'Smoothed Distribution of {col}', color='orange')
            
            plt.title(f'Distribution of {col} Across Dataframes')
            plt.xlabel('Bins')
            plt.ylabel('Value')
            plt.legend()
            plt.show()
            # plt.savefig('output_plot.png')


def viz(json_path):
    """
    Main core function to visualize distributions.
    $ python viz.py --folder_path /Users/surajkwork/Documents/Projects/ProteinLigand/path_to_json

    """
    # Get attributes from JSON file

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The file {json_path} does not exist.")

    # Load the JSON file
    with open(json_path, 'r') as file:
        config = json.load(file)

    path_to_PDB_files = config.get("path_to_PDB_files")
    box_size = config.get("box_size")
    ligand_name = config.get("ligand_name")
    ligand_path = config.get("ligand_path")
    columns = config.get("columns")
    group_ids = config.get("group_ids")
    smooth_distribution = config.get("smooth_distribution")
    smooth_tolerance = config.get("smooth_tolerance")
    bin_size = config.get("bin_size")
    n_bins = config.get("n_bins")
    range_values = config.get("range")
    merge_distributions = config.get("merge_distributions")
    sliding_window = config.get("sliding_window")

    try:
        process_dataframes(path_to_PDB_files, box_size, ligand_name, ligand_path, columns, smooth_distribution, smooth_tolerance, bin_size, n_bins, merge_distributions, group_ids, range_values, sliding_window)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize distributions from folder")
    parser.add_argument('--json_path', type=Path, help="Path to the folder containing the JSON files")
    args = parser.parse_args()
    print(f"Folder Path: {args.json_path}") 
    viz(args.json_path)
