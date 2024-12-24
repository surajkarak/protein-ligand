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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
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

def residue_overlap_distance(site1, site2, distancetype):
    residues1 = set(site1.get('residues', []))
    residues2 = set(site2.get('residues', []))
    
    if not residues1 or not residues2: 
        return 1.0 
    
    # Jaccard distance = 1 - Jaccard similarity
    return 1 - len(residues1 & residues2) / len(residues1 | residues2)

def pairwise_distances_with_library(sites, distance_func, distancetype):
    def distance_wrapper(i, j):
        return distance_func(sites[int(i[0])], sites[int(j[0])], distancetype)
    
    indices = list(range(len(sites)))
    condensed_matrix = pdist([[i] for i in indices], metric=lambda i, j: distance_wrapper(i, j))
    
    return squareform(condensed_matrix)

subset = all_binding_sites[:100]  # First 100 sites
distance_matrix = pairwise_distances_with_library(subset, residue_overlap_distance)

# Based on residue scores

def residue_score_distance(site1, site2, distancetype):
    res_scores_1 = site1.get('residue_scores', {})
    res_scores_2 = site2.get('residue_scores', {})
    rnames = list(set.union(set(res_scores_1.keys()), set(res_scores_2.keys())))
    res_scores_1 = {**dict.fromkeys(rnames, 0), **res_scores_1}
    res_scores_2 = {**dict.fromkeys(rnames, 0), **res_scores_2}
    res_scores_1 = np.asarray(list(res_scores_1.values()), np.float32)
    res_scores_2 = np.asarray(list(res_scores_2.values()), np.float32)
    
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

subset = all_binding_sites[:100]
distance_matrix_scores = pairwise_distances_with_library(subset, residue_score_distance, distancetype)

# Based on distance vectors

# def distance_vectors(binding sites):

# Step 2: Clustering
# Output to include 2D t-SNE visualizations and cluster labels for binding sites

# Agglomerative

def agglom_cluster(distance_matrix):
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
    agglom_model = AgglomerativeClustering(n_clusters=13, metric='precomputed', linkage='average')
    cluster_labels = agglom_model.fit_predict(distance_matrix)
    return cluster_labels


# MeanShift
def meanshift_cluster(distance_matrix):
    mean_shift = MeanShift()
    mean_shift_labels = mean_shift.fit_predict(distance_matrix)
    return mean_shift_labels

# DBSCAN
def dbscan_cluster(distance_matrix):
    dbscan = DBSCAN(metric='precomputed', eps=0.5, min_samples=2)
    dbscan_labels = dbscan.fit_predict(distance_matrix)
    return dbscan_labels

# OPTICS
def optics_cluster(distance_matrix):
    optics = OPTICS(metric='precomputed', min_samples=5, xi=0.05, min_cluster_size=0.1)
    optics_labels = optics.fit_predict(distance_matrix)
    return optics_labels

# Step 3: Output of clusters and binding sites mapped to clusters




