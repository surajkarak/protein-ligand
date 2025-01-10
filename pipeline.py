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
from ordered_set import OrderedSet


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


# Based on residue scores

def residue_score_distance(site1, site2, distancetype):
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
    elif distancetype =="L1":
        dist = np.sum(np.abs(res_scores_1 - res_scores_2))
    elif distancetype =="Jaccard":
        min_sum = np.sum(np.minimum(res_scores_1, res_scores_2))
        max_sum = np.sum(np.maximum(res_scores_1, res_scores_2))
        sim = min_sum / max_sum if max_sum != 0 else 0
        dist = 1 - sim
    
    return dist


# Based on distance vectors

# def distance_vectors(binding sites):

# def get_residue_coordinates(binding_site, target_data):
#     residues = binding_site.get('residues', [])
#     target_residues = {
#         (chain, res_id, res_name): coord 
#         for chain, res_id, res_name, coord in zip(
#             target_data.get('chain_ids', []),
#             target_data.get('res_ids', []),
#             target_data.get('res_names', []),
#             target_data.get('coords', [])
#         )
#     }
    
#     # Extract coordinates for residues in the binding site
#     residue_coords = []
#     for residue in residues:
#         chain_id, res_id, res_name = residue.split('_')  # Assuming residue is in "chain_resid_name" format
#         coord = target_residues.get((chain_id, int(res_id), res_name))
#         if coord is not None:
#             residue_coords.append(coord)
#     return np.array(residue_coords)

def get_atoms_in_binding_site(binding_site): #helper function to get atoms within a binding site
    """
    Extract atoms that belong to the residues in a given binding site.

    Parameters:
    - binding_site: dict, a single binding site from all_binding_sites.
    - target_data: dict, the corresponding target data from all_target_data.

    Returns:
    - List of atoms with their details for the binding site.
    """

    file_name = binding_site['file']
    target= [entry for entry in all_target_data if entry['file'] in file_name]
    target_data = next((entry for entry in target if entry['file'] == file_name), None)
    binding_site_residues = binding_site['site']['residues']


    parsed_residues = []
    for residue in binding_site_residues:
        chain_id, res_id, res_name = residue.split('_')
        parsed_residues.append((chain_id, int(res_id), res_name))

    
    chain_ids = target_data['target']['chain_ids']
    res_ids = target_data['target']['res_ids']
    res_names = target_data['target']['res_names']
    atom_names = target_data['target']['atom_names']
    coords = target_data['target']['coords']
    elements = target_data['target']['elements']

    atoms_in_site = []
    for chain_id, res_id, res_name in parsed_residues:
        # Find indices matching the residue
        mask = (chain_ids == chain_id) & (res_ids == res_id) & (res_names == res_name)
        matching_indices = np.where(mask)[0]

        # Extract atom details for matching residues
        for idx in matching_indices:
            atoms_in_site.append({
                'chain_id': chain_id,
                'res_id': res_id,
                'res_name': res_name,
                'atom_name': atom_names[idx],
                'coords': coords[idx].tolist(),
                'element': elements[idx],
            })

    return atoms_in_site

def atoms_of_interest(binding_sites):
    atoms_of_interest = []
    for site in binding_sites:
        atoms = get_atoms_in_binding_site(site) 
        atoms_of_interest.append(atoms)
    all_atoms = [
        f"{atom['chain_id']}_{atom['res_id']}_{atom['res_name']}_{atom['atom_name']}" 
        for atom_list in atoms_of_interest 
        for atom in atom_list
        ]
    unique_atoms = OrderedSet(all_atoms)
    atoms_of_interest_flat = [atom for atoms in atoms_of_interest for atom in atoms]
    print("Set of atoms of interest created (Distance Vector Method)") 
    return unique_atoms, atoms_of_interest_flat 

def get_atom_coordinates(atom_data):
   
    chain_id, res_id, res_name, atom_name = atom_data
    for entry in atoms_of_interest_flat:  # Assuming this contains atom data dictionaries
        if (
            entry['chain_id'] == chain_id and
            entry['res_id'] == int(res_id) and
            entry['res_name'] == res_name and
            entry['atom_name'] == atom_name
        ):
            return entry['coords']
    raise ValueError(f"Coordinates not found for atom: {atom_data}")

def binding_site_vector(binding_site, unique_atoms):
    """
    Calculate the vector for a binding site based on the distances between hotspots and unique atoms.

    Parameters:
        binding_site (dict): The binding site data containing hotspots and residues.
        unique_atoms (set): A set of unique atom identifiers.

    Returns:
        list: The binding site vector with minimum distances to hotspots for each atom.
    """
    # Initialize a dummy vector for all unique atoms with a large default value (e.g., 10.0)
    dummy_vector = {atom: 10.0 for atom in unique_atoms}

    # Get the set of residues in the current binding site
    residues_in_site = set(binding_site['site']['residues'])  # Example: ['A_14_GLY', 'A_15_VAL', ...]

    # Map residues to atoms using atoms_of_interest_flat
    atoms_in_site = set(
        f"{atom['chain_id']}_{atom['res_id']}_{atom['res_name']}_{atom['atom_name']}"
        for atom in atoms_of_interest_flat
        if f"{atom['chain_id']}_{atom['res_id']}_{atom['res_name']}" in residues_in_site
    )

    # Create a list to hold all hotspot vectors
    hotspot_vectors = []

    # Iterate through each hotspot in the binding site
    for hotspot in binding_site['site']['hotspots']:
        hotspot_coords = hotspot['center']  # Coordinates of the hotspot
        hotspot_vector = dummy_vector.copy()  # Initialize a new vector for this hotspot

        # Iterate through atoms in the binding site that are also in unique_atoms
        for atom in unique_atoms.intersection(atoms_in_site):
            # Extract the atom's details and coordinates
            atom_data = atom.split('_')
            atom_coords = get_atom_coordinates(atom_data)  # Define this helper function to fetch atom coords

            # Calculate the Euclidean distance between the hotspot and the atom
            distance = np.linalg.norm(np.array(hotspot_coords) - np.array(atom_coords))

            # Update the hotspot vector with the calculated distance
            hotspot_vector[atom] = distance

        # Add the hotspot vector to the list of vectors
        hotspot_vectors.append(hotspot_vector)

    # Calculate the binding site vector as the minimum distance across all hotspots for each atom
    binding_site_vector = [
        min(hotspot_vector[atom] for hotspot_vector in hotspot_vectors) for atom in unique_atoms # Ordered set in python. To check.
    ]

    return binding_site_vector

def distance_between_vectors(binding_site_1, binding_site_2, distancetype):

    binding_site_vector1 = binding_site_vector(binding_site_1, unique_atoms)
    binding_site_vector2 = binding_site_vector(binding_site_2, unique_atoms)

    if distancetype == "euclidean":
        distance = np.linalg.norm(np.array(binding_site_vector1) - np.array(binding_site_vector2))
        # sim = 1 - dist
    elif distancetype =="L1":
        distance = np.linalg.norm(np.array(binding_site_vector1) - np.array(binding_site_vector2), ord=1)
    elif distancetype == "jaccard":
        min_sum = np.sum(np.minimum(np.array(binding_site_vector1), np.array(binding_site_vector2)))
        max_sum = np.sum(np.maximum(np.array(binding_site_vector1), np.array(binding_site_vector2)))
        similarity = min_sum / max_sum if max_sum != 0 else 0  # Handle divide-by-zero
        distance = 1 - similarity

    return distance


# subset = all_binding_sites[:100]  # First 100 sites
# distance_matrix = pairwise_distances_with_library(subset, distance_vector)

def pairwise_distances_with_library(sites, distance_func, distance_type):
    def distance_wrapper(i, j, distance_type):
        return distance_func(sites[int(i[0])], sites[int(j[0])], distance_type)
    
    indices = list(range(len(sites)))
    condensed_matrix = pdist([[i] for i in indices], metric=lambda i, j: distance_wrapper(i, j, distance_type))
    
    return squareform(condensed_matrix)

# Step 2: Clustering

def perform_clustering(distance_matrix, clustering_args):
    """Perform clustering based on the specified clustering model."""

    clustering_model = clustering_args.get("type")
    clustering_params = clustering_args.get("parameters")

    #Agglomerative 
    def agglomerative(distance_matrix, clustering_params):
        linkage_method = clustering_params.get("method")
        linkage_matrix = linkage(squareform(distance_matrix), method= linkage_method)
        fcluster_threshold = clustering_params.get("threshold")
        fcluster_criterion = clustering_params.get("criterion")
        clusters = fcluster(linkage_matrix, t=fcluster_threshold, criterion=fcluster_criterion)
        agglom_model = AgglomerativeClustering(n_clusters=len(clusters), metric='precomputed', linkage=linkage_method)
        cluster_labels = agglom_model.fit_predict(distance_matrix)
        embedding = TSNE(n_components=2, metric='precomputed', init='random').fit_transform(distance_matrix)
        plt.figure(figsize=(10, 6))
        for cluster_id in np.unique(cluster_labels):
            cluster_points = embedding[cluster_labels == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', s=50)
        plt.title('Clusters Visualized Using t-SNE')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.show()
        return cluster_labels
    
    # Meanshift
    def mean_shift(distance_matrix):
        mean_shift = MeanShift()
        mean_shift_labels = mean_shift.fit_predict(distance_matrix)
        embedding = TSNE(n_components=2, metric='precomputed', init='random').fit_transform(distance_matrix)
        plt.figure(figsize=(10, 6))
        for cluster_id in np.unique(mean_shift_labels):
            cluster_points = embedding[mean_shift_labels == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', s=50)

        plt.title('Clusters Visualized Using t-SNE')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.show()
        return mean_shift_labels
    
    # DBSCAN
    def dbscan(distance_matrix, clustering_params):
        epsilon = clustering_params.get("eps")
        minimum_samples = clustering_params.get("min_samples")
        dbscan = DBSCAN(metric='precomputed', eps=epsilon, min_samples=minimum_samples)
        dbscan_labels = dbscan.fit_predict(distance_matrix)
        embedding = TSNE(n_components=2, metric='precomputed', init='random').fit_transform(distance_matrix)
        embedding = TSNE(n_components=2, metric='precomputed', init='random').fit_transform(distance_matrix)
        # Scatter plot
        plt.figure(figsize=(8, 6))
        for cluster in set(dbscan_labels):
            cluster_points = embedding[dbscan_labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
        plt.title('Binding Site Clusters (t-SNE)')
        plt.legend()
        plt.show()
        return dbscan_labels

    # OPTICS
    def optics(distance_matrix, clustering_params):
        minimum_samples = clustering_params.get("min_samples")
        xi = clustering_params.get("xi")
        minimum_cluster_size = clustering_params.get("min_cluster_size")
        optics = OPTICS(metric='precomputed', min_samples=5, xi=xi, min_cluster_size=minimum_cluster_size)
        optics_labels = optics.fit_predict(distance_matrix)
        embedding = TSNE(n_components=2, metric='precomputed', init='random').fit_transform(distance_matrix)
        # Scatter plot
        plt.figure(figsize=(8, 6))
        for cluster in set(optics_labels):
            cluster_points = embedding[optics_labels == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
        plt.title('Binding Site Clusters (t-SNE)')
        plt.legend()
        plt.show()
        return optics_labels

    if clustering_model == 'agglomerative':
        cluster_labels = agglomerative(distance_matrix, clustering_params)
        return cluster_labels        
    elif clustering_model == 'meanshift':
        mean_shift_labels = mean_shift(distance_matrix)
        return mean_shift_labels
    elif clustering_model == 'dbscan':
        dbscan_labels = dbscan(distance_matrix, clustering_params)
        return dbscan_labels
    elif clustering_model == 'optics':
        optics_labels = optics(distance_matrix, clustering_params)
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

    print(f"Binding sites with cluster labels saved to {output_file}")




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

    global subset
    folder_path =  config.get("path_to_pkl_files") 
    subset = config.get("subset_to_analyse")
    distance_args = config.get("distance_metric")
    distance_metric = distance_args.get("metric_type")
    distance_type = distance_args.get("distance_type")  
    clustering_args = config.get("clustering_model")
    clustering_model = clustering_args.get("type")

    global all_target_data

    all_binding_sites, all_target_data = extraction(folder_path)
    print("Extraction completed")
    binding_sites = all_binding_sites[:subset]

    
    def calculate_distance_matrix(binding_sites, distance_metric, distance_type):
        """Calculate the pairwise distance matrix based on the specified distance metric."""
        if distance_metric == 'residue_overlap':
            return pairwise_distances_with_library(binding_sites, residue_overlap_distance, distance_type)
        elif distance_metric == 'residue_score':
            return pairwise_distances_with_library(binding_sites, residue_score_distance, distance_type)
        elif distance_metric == 'distance_vector':
            global unique_atoms, atoms_of_interest_flat
            unique_atoms, atoms_of_interest_flat = atoms_of_interest(binding_sites)
            return pairwise_distances_with_library(binding_sites, distance_between_vectors, distance_type)
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

    distance_matrix = calculate_distance_matrix(binding_sites, distance_metric, distance_type)
    print(f"Distance matrix calculated using {distance_metric}")
    labels = perform_clustering(distance_matrix, clustering_args)
    print(f"Clustering done using: {clustering_model}")
    mapping(binding_sites, labels)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster binding sites from PKL files (frames)")
    parser.add_argument('--json_path', type=Path, help="Path to the folder containing the JSON files with the parameters")
    args = parser.parse_args()
    print(f"Folder Path: {args.json_path}") 
    cluster(args.json_path)
