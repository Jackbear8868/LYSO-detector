#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================
Created on 2025.04.07
@author: oyang
====================================
"""
import uproot
import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition  import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, silhouette_score
from sklearn.metrics import adjusted_rand_score
import hdbscan
from hdbscan.prediction import approximate_predict

#======================================
# read LYSO's leaves
#======================================
def get_data():

    fin = "LYSO4NTUMentor.root"
    fTree = "LYSO"
    LYSO = uproot.open(fin)[fTree]
    print ("Read ROOT file:", fin)

    s1 = ak.to_numpy(LYSO.arrays()['S1'])
    s2 = ak.to_numpy(LYSO.arrays()['S2'])
    s3 = ak.to_numpy(LYSO.arrays()['S3'])
    s4 = ak.to_numpy(LYSO.arrays()['S4'])
    ID = ak.to_numpy(LYSO.arrays()['ID'])
    event = ak.to_numpy(LYSO.arrays()['EVENT'])
    # y = ak.to_numpy(LYSO.arrays()['y'])
    LYSO.close()

    type_arr = np.zeros(100, dtype=int)
    for id in ID:
        id = id.astype(int)
        if id > 100:
            print("not found index: ", id)
        type_arr[id] += 1

    print("ID type count:", np.count_nonzero(type_arr)) # 20 types in this
    print("ID: ", type_arr)
    return s1, s2, s3, s4, ID.astype(int)

# PCA + StandardScaler
def data_preprocessing(s1, s2, s3, s4, n_components=2):

    s1 = np.expand_dims(s1, axis=1)
    s2 = np.expand_dims(s2, axis=1)
    s3 = np.expand_dims(s3, axis=1)
    s4 = np.expand_dims(s4, axis=1)

    s_vector = np.concat((s1, s2, s3, s4), axis=1)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(s_vector)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    return X_pca, X_std

def run_kmeans(X, X_pca, n_cluster):

    kmeans = KMeans(n_clusters=n_cluster, random_state=42).fit(X)
    labels = kmeans.labels_ 

    plt.figure(figsize=(6,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', s=5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('KMeans (k={})'.format(n_cluster))
    plt.savefig("./pic/KMeans_{}.png".format(n_cluster))
    return labels

def run_spectral(X, X_pca, n_clusters, n_neighbors):
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        eigen_solver='arpack',
        assign_labels='kmeans',
        random_state=42
    )
    labels = spectral.fit_predict(X)
    plt.figure(figsize=(6,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab20', s=5)
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.title('Spectral Clustering (k={})'.format(n_clusters))
    plt.savefig("./pic/Spectral_{}.png".format(n_clusters))
    return labels

def run_hdbscan(X, X_pca, min_cluster_size, min_samples=None):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples, prediction_data=True
    )
    labels = clusterer.fit_predict(X)    

    plt.figure(figsize=(6,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab20', s=5)
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.title('HDBSCAN (min_cluster_size={})'.format(min_cluster_size))
    plt.savefig("./pic/HDBSCAN_{}.png".format(min_cluster_size))
    return labels

def evaluate_clustering(y_true, y_pred, method_name=""):
     
    print(f"\n=== {method_name} Evaluation ===")
    h = homogeneity_score(y_true, y_pred)
    v = v_measure_score(y_true, y_pred)
    c = completeness_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    print(f"Homogeneity = {h:.3f}, Completeness = {c:.3f}, V-measure = {v:.3f}")
    print("Adjusted Rand Index =", ari)


if __name__ == "__main__":

    cluster_num = 11
    s1, s2, s3, s4, y_true = get_data()
    X_pca, X_std = data_preprocessing(s1, s2, s3, s4, n_components=2)

    if not os.path.exists("./pic"):
        os.makedirs("./pic")

    X = X_pca
    # KMeans
    y_pred = run_kmeans(X, X_pca, n_cluster=cluster_num)
    evaluate_clustering(y_true, y_pred, method_name="KMeans")

    # spectral clustering
    y_spec = run_spectral(X_std, X_pca, n_clusters=cluster_num, n_neighbors=40)
    evaluate_clustering(y_true, y_spec, "Spectral Clustering")

    # HDBSCAN
    y_hdb = run_hdbscan(X_std, X_pca, min_cluster_size=40)
    evaluate_clustering(y_true, y_hdb, "HDBSCAN")

    unique_labels, counts = np.unique(y_hdb, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Type {label}: {count} samples")

    noise_idx = np.where(y_hdb == -1)[0]
    if noise_idx.size > 0:
        # compute centroids from non-noise assignments
        centroids = np.vstack([
            X_std[y_hdb == cid].mean(axis=0)
            for cid in range(cluster_num)
        ])
        X_noise = X_std[noise_idx]
    
        distances = np.linalg.norm(X_noise[:, None, :] - centroids[None, :, :], axis=2)
        assigned = distances.argmin(axis=1)

        y_hdb_extended = y_hdb.copy()
        y_hdb_extended[noise_idx] = assigned

        plt.figure(figsize=(6,6))
        plt.scatter(X_pca[:,0], X_pca[:,1], c=y_hdb_extended, cmap='tab20', s=5)
        plt.xlabel('PC1'); plt.ylabel('PC2')
        plt.title('HDBSCAN + Noise Assignment')
        plt.savefig("./pic/HDBSCAN_extended.png")

        evaluate_clustering(y_true, y_hdb_extended, "HDBSCAN + Noise Assignment")
    else:
        print("No noise points to assign.")

