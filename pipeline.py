import argparse
from pathlib import Path
import pandas as pd
import json
import random
from collections import defaultdict
import os
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE



# folder_path = '/Users/surajkwork/Documents/Projects/ProteinLigand/protein-ligand/protein-ligand/BindingSiteAnalysis/kras_md_sites_1'   
# all_binding_sites = []


def extraction(folder_path):
    all_binding_sites = []
    all_target_data = []

    for file_name in sorted(os.listdir(folder_path)):   
        if file_name.endswith('.pkl'):   
            file_path = os.path.join(folder_path, file_name)
            
            try:
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                
                sites = data.get('sites', [])
                for site in sites:
                    all_binding_sites.append({'file': file_name, 'site': site})  
            
                target = data.get('target', {})
                if target:
                    all_target_data.append({'file': file_name, 'target': target}) 
            
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    return all_binding_sites, all_target_data
    # print(f"Total target data entries extracted: {len(all_target_data)}")


# Step 1: Distance/similarity calculation

# Based on residue set intersections

def residue_overlap_distance(site1, site2, norm_by):
    residues1 = set(site1["site"].get('residues', []))
    residues2 = set(site2["site"].get('residues', []))
    
    if not residues1 or not residues2: 
        return 1.0 
    
    inter = len(residues1 & residues2)
    union = len(residues1 | residues2)

    # norm by average
    if norm_by == "average":
        sim = 2 * inter / (len(residues1) * len(residues2))
    # norm by max
    if norm_by == "max":
        sim = inter / max(len(residues1), len(residues2))
    # norm by min
    if norm_by == "min":
        sim = inter / min(len(residues1), len(residues2))
    # norm by union
    if norm_by == "union":
        sim = inter / union
    # distance can be calculated as inverse
    dist = 1 - sim
    return dist

    # # Jaccard distance = 1 - Jaccard similarity
    # return 1 - len(residues1 & residues2) / len(residues1 | residues2)

def pairwise_distances_with_library(sites, distance_func, norm_by):
    def distance_wrapper(i, j, norm_by):
        return distance_func(sites[int(i[0])], sites[int(j[0])], norm_by)
    
    indices = list(range(len(sites)))
    condensed_matrix = pdist([[i] for i in indices], metric=lambda i, j: distance_wrapper(i, j, norm_by))
    
    return squareform(condensed_matrix)


# Based on residue scores

def residue_score_distance(site1, site2, distancetype ):
    res_scores_1 = site1["site"].get('residue_scores', {})
    res_scores_2 = site2["site"].get('residue_scores', {})
    rnames = list(set.union(set(res_scores_1.keys()), set(res_scores_2.keys())))
    res_scores_1 = {**dict.fromkeys(rnames, 0), **res_scores_1}
    res_scores_2 = {**dict.fromkeys(rnames, 0), **res_scores_2}
    res_scores_1 = np.asarray(list(res_scores_1.values()), np.float32)
    res_scores_2 = np.asarray(list(res_scores_2.values()), np.float32)
    # dist = np.sqrt(np.sum((res_scores_1 - res_scores_2) ** 2))

    if distancetype == "euclidean":
        dist = np.sqrt(np.sum((res_scores_1 - res_scores_2) ** 2))
        sim = 1 - dist
    if distancetype =="L1":
        dist = np.sum(np.abs(res_scores_1 - res_scores_2))
    if distancetype =="Jaccard":
        min_sum = np.sum(np.minimum(res_scores_1, res_scores_2))
        max_sum = np.sum(np.maximum(res_scores_1, res_scores_2))
        sim = min_sum / max_sum
        dist = 1 - sim
    
    return dist


# Based on distance vectors

# def distance_vectors(binding sites):

def get_residue_coordinates(binding_site, target_data):
    residues = binding_site.get('residues', [])
    target_residues = {
        (chain, res_id, res_name): coord 
        for chain, res_id, res_name, coord in zip(
            target_data.get('chain_ids', []),
            target_data.get('res_ids', []),
            target_data.get('res_names', []),
            target_data.get('coords', [])
        )
    }
    
    # Extract coordinates for residues in the binding site
    residue_coords = []
    for residue in residues:
        chain_id, res_id, res_name = residue.split('_')  # Assuming residue is in "chain_resid_name" format
        coord = target_residues.get((chain_id, int(res_id), res_name))
        if coord is not None:
            residue_coords.append(coord)
    return np.array(residue_coords)


def hotspot_to_residue(binding_site_entry):
    file_name = binding_site_entry['file']  # Identify the file source
    binding_site = binding_site_entry['site']
    subset_target_data = [entry for entry in all_target_data if entry['file'] in file_name]
    target_entry = next((entry for entry in subset_target_data if entry['file'] == file_name), None)

    # Extract residue coordinates
    residue_coords = get_residue_coordinates(binding_site, target_entry['target'])

    # Extract hotspot coordinates
    hotspots = binding_site.get('hotspots', [])
    hotspot_coords = np.array([hotspot['center'] for hotspot in hotspots])

    if residue_coords.size == 0 or hotspot_coords.size == 0:
        return None  # No valid distances if either is empty

    # Compute pairwise distances (rows: hotspots, columns: residues)
    pairwise_distances = cdist(residue_coords, hotspot_coords, metric="euclidean")
    min_distances = np.min(pairwise_distances, axis=1)
    max_dist = 8.
    vectors = 1 - np.clip(min_distances / max_dist, 0, 1)
    return vectors

def distance_vector(site1, site2, distancetype):
    d1 = hotspot_to_residue(site1)
    d2 = hotspot_to_residue(site2)
    max_length = max(len(d1), len(d2))
    vector1_padded = np.pad(d1, (0, max_length - len(d1)), mode='constant')
    vector2_padded = np.pad(d2, (0, max_length - len(d2)), mode='constant')

    if distancetype == "euclidean":
        distance = np.linalg.norm(vector1_padded - vector2_padded)
        # sim = 1 - dist
    if distancetype =="L1":
        distance = np.linalg.norm(vector1_padded - vector2_padded, ord=1)


    return distance


# subset = all_binding_sites[:100]  # First 100 sites
# distance_matrix = pairwise_distances_with_library(subset, distance_vector)


    

# Step 2: Clustering

def perform_clustering(distance_matrix, clustering_model):
    """Perform clustering based on the specified clustering model."""
    if clustering_model == 'agglomerative':
        linkage_matrix = linkage(squareform(distance_matrix), method='average')
        clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
        agglom_model = AgglomerativeClustering(n_clusters=len(clusters), metric='precomputed', linkage='average')
        cluster_labels = agglom_model.fit_predict(distance_matrix)
        return cluster_labels
    elif clustering_model == 'meanshift':
        mean_shift = MeanShift()
        mean_shift_labels = mean_shift.fit_predict(distance_matrix)
        return mean_shift_labels
    elif clustering_model == 'dbscan':
        dbscan = DBSCAN(metric='precomputed', eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(distance_matrix)
        return dbscan_labels
    elif clustering_model == 'optics':
        optics = OPTICS(metric='precomputed', min_samples=5, xi=0.05, min_cluster_size=0.1)
        optics_labels = optics.fit_predict(distance_matrix)
        return optics_labels
    else:
        raise ValueError(f"Unsupported clustering model: {clustering_model}")
    

# Step 3: Output of clusters and binding sites mapped to clusters

def mapping(subset, labels):
    labels = [int(label) for label in labels]
    subset_with_labels = [
        {"residues": site["site"].get("residues", []), "cluster_label": label}
        for site, label in zip(subset, labels)
    ]

    output_file = "bindingsites_with_labels.json"
    with open(output_file, "w") as f:
        json.dump(subset_with_labels, f, indent=4)

    print(f"Binging sites with cluster labels saved to {output_file}")




def cluster(json_path):
    """
    Main core function to execute the clustering
    $ python viz.py --folder_path /Users/surajkwork/Documents/Projects/ProteinLigand/path_to_json

    """
    # Get attributes from JSON file

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The file {json_path} does not exist.")

    # Load the JSON file
    with open(json_path, 'r') as file:
        config = json.load(file)

    folder_path =  config.get("path_to_pkl_files") 
    subset = config.get("subset_to_analyse")
    distance_metric = config.get("distance_metric")
    clustering_model = config.get("clustering_model")
    # "distance_type": "euclidean"

    global all_target_data

    all_binding_sites, all_target_data = extraction(folder_path)
    print("Extraction completed")
    binding_sites = all_binding_sites[:subset]

    
    def calculate_distance_matrix(binding_sites, distance_metric):
        """Calculate the pairwise distance matrix based on the specified distance metric."""
        if distance_metric == 'residue_overlap':
            return pairwise_distances_with_library(binding_sites, residue_overlap_distance )
        elif distance_metric == 'residue_score':
            return pairwise_distances_with_library(binding_sites, residue_score_distance)
        elif distance_metric == 'distance_vector':
            return pairwise_distances_with_library(binding_sites, distance_vector)
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

    distance_matrix = calculate_distance_matrix(binding_sites, distance_metric)
    print(f"Distance matrix calculated using {distance_metric}")
    labels = perform_clustering(distance_matrix, clustering_model)
    print(f"Clustering done using: {clustering_model}")
    mapping(binding_sites, labels)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster binding sites from PKL files (frames)")
    parser.add_argument('--json_path', type=Path, help="Path to the folder containing the JSON files with the parameters")
    args = parser.parse_args()
    print(f"Folder Path: {args.json_path}") 
    cluster(args.json_path)
