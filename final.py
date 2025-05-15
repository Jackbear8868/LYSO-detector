from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score,pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import numpy as np
import pandas as pd
import hdbscan

def score(true_labels, predicted_labels):
    nmi = normalized_mutual_info_score(true_labels,predicted_labels)
    print(f"NMI: {nmi:.4f}")

# Centroid-based
def kmeans(data, clusters):
    model = KMeans(n_clusters=clusters, init="k-means++", n_init=10, max_iter= 10000,random_state=0)
    model.fit(data)
    labels = model.labels_
    score(ground_truth, labels)


def kmedoids(data, clusters):
    pass

# Density-based
def dbscan(data):
    db = DBSCAN(eps=1, min_samples=10)
    labels = db.fit_predict(data)
    score(ground_truth, labels)


def hdb(data):
    hdb = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = hdb.fit_predict(data)
    score(ground_truth, labels)

def optics(data):
    optics = OPTICS(min_samples=5, xi=0.03, min_cluster_size=0.03)
    optics.fit(data)
    labels = optics.labels_
    score(ground_truth, labels)


# Hierarchical-based
def agglomerative(data, clusters):
    model = AgglomerativeClustering(n_clusters=clusters, linkage='ward') # ward, complete, average, single
    labels = model.fit_predict(data)
    score(ground_truth,labels)

def divisive():
    pass


# Model-based
def gaussian(data, clusters): # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=clusters, covariance_type='full', random_state=0) # 'full', 'tied', 'diag', 'spherical'
    gmm.fit(data)
    labels = gmm.predict(data)
    score(ground_truth, labels)


# Graph-based
def spectral(data, clusters):
    sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans')
    labels = sc.fit_predict(data)
    score(ground_truth, labels)
    

def mean_shift(data):
    bandwidth = estimate_bandwidth(data, quantile=0.05, n_samples=50)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    score(ground_truth, labels)


def affinity(data):
    similarity = -pairwise_distances(data, metric='euclidean')  # 注意要加負號
    model = AffinityPropagation(affinity='precomputed', damping=0.9)
    model.fit(similarity)
    labels = model.labels_
    score(ground_truth, labels)

def som(data, clusters): # Self-Organizing Maps
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Step 2: 初始化 SOM（選擇一個合理的網格大小）
    som_x = som_y = int(np.ceil(np.sqrt(clusters)))  # ex: 9 clusters -> 3x3 SOM
    som = MiniSom(x=som_x, y=som_y, input_len=data.shape[1], sigma=1.2, learning_rate=0.2)
    som.random_weights_init(data_scaled)
    som.train(data_scaled, num_iteration=10000, verbose=False)

    # Step 3: 取得每個資料點的 BMU (Best Matching Unit)
    winner_coordinates = np.array([som.winner(d) for d in data_scaled])
    # 把 2D 座標攤平成 1D 編號，例如 (2,1) -> 2*som_y + 1
    flattened_indices = np.ravel_multi_index(winner_coordinates.T, dims=(som_x, som_y))

    # Step 4: 用 KMeans 對 SOM 的 BMU 結果做最終分群（這是常見做法）
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flattened_indices.reshape(-1, 1))
    labels = kmeans.labels_

    # Step 5: 回傳分群結果
    score(ground_truth, labels)

    

data = pd.read_csv('public_data.csv')
solution = pd.read_csv('solution.csv')
ground_truth = solution['class'].values
clusters = 15

# kmeans(data,clusters)
# dbscan(data)
# hdb(data)
# optics(data)
# agglomerative(data, clusters)
# gaussian(data, clusters)
# spectral(data, clusters)
# mean_shift(data)
# affinity(data)
som(data, clusters)
