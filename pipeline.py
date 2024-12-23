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
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

folder_path = '/Users/surajkwork/Documents/Projects/ProteinLigand/protein-ligand/protein-ligand/BindingSiteAnalysis/kras_md_sites_1'   
all_binding_sites = []
all_target_data = []

def extraction(folder_path):

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

    # print(f"Total binding sites extracted: {len(all_binding_sites)}")
    # print(f"Total target data entries extracted: {len(all_target_data)}")


# Step 1: Distance/similarity calculation

# Based on residue set intersections

def residue_overlap_distance(site1, site2):
    residues1 = set(site1.get('residues', []))
    residues2 = set(site2.get('residues', []))
    
    if not residues1 or not residues2: 
        return 1.0 
    
    # Jaccard distance = 1 - Jaccard similarity
    return 1 - len(residues1 & residues2) / len(residues1 | residues2)

def pairwise_distances_with_library(sites, distance_func):
    def distance_wrapper(i, j):
        return distance_func(sites[int(i[0])], sites[int(j[0])])
    
    indices = list(range(len(sites)))
    condensed_matrix = pdist([[i] for i in indices], metric=lambda i, j: distance_wrapper(i, j))
    
    return squareform(condensed_matrix)

subset = all_binding_sites[:100]  # First 100 sites
distance_matrix = pairwise_distances_with_library(subset, residue_overlap_distance)

# Based on residue scores

def residue_score_distance(site1, site2):
    res_scores_1 = site1.get('residue_scores', {})
    res_scores_2 = site2.get('residue_scores', {})
    rnames = list(set.union(set(res_scores_1.keys()), set(res_scores_2.keys())))
    res_scores_1 = {**dict.fromkeys(rnames, 0), **res_scores_1}
    res_scores_2 = {**dict.fromkeys(rnames, 0), **res_scores_2}
    res_scores_1 = np.asarray(list(res_scores_1.values()), np.float32)
    res_scores_2 = np.asarray(list(res_scores_2.values()), np.float32)
    # euclidean distance
    dist = np.sqrt(np.sum((res_scores_1 - res_scores_2) ** 2))
    sim = 1 - dist
    # # L1 distance
    # dist = np.sum(np.abs(res_scores_1 - res_scores_2))
    # # Jaccard index
    # min_sum = np.sum(np.minimum(res_scores_1, res_scores_2))
    # max_sum = np.sum(np.maximum(res_scores_1, res_scores_2))
    # sim = min_sum / max_sum
    # dist = 1 - sim
    
    return dist

subset = all_binding_sites[:100]
distance_matrix_scores = pairwise_distances_with_library(subset, residue_score_distance)

# BAsed on distance vectors

# def distance_vectors(binding sites):

# Step 2: Clustering
# Output to include 2D t-SNE visualizations and cluster labels for binding sites

# Agglomerative

# MeanShift

# DBSCAN

# OPTICS





