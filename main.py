import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pywt
import statistics as stats
from kmeans import KMeans
from gmm import GMM
from dbscan import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
results_kmeans = []
results_dbscan = []
results_gmm = []

def Get_Feature(picture, cortes):
    LL = picture
    for i in range(cortes):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    return LL.flatten()

def load_data():
    image_paths = []
    labels = []
    data = []
    # Load the images and labels from the folders
    for emotion in emotions:
        folder_path = os.path.join(path_file, emotion)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                image_path = os.path.join(folder_path, file_name)
                image_paths.append(image_path)
                labels.append(emotion)

    for image_path in image_paths:
        image = plt.imread(image_path)
        feature_vector = Get_Feature(image, cortes=3)  # Use Get_Feature function with cortes=3
        data.append(feature_vector)
    
    data = np.array(data)
    labels = np.array(labels)

    # Normalize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, labels

# Calculate purity of clusters
def purity(clusters, labels):
    cluster_labels = np.unique(clusters)
    total = 0
    for label in cluster_labels:
        cluster_indices = np.where(clusters == label)[0]
        cluster_emotions = labels[cluster_indices]
        unique_emotions, emotion_counts = np.unique(cluster_emotions, return_counts=True)
        total += np.max(emotion_counts)
    return total / len(labels)

def execute_kmeans(data, k, umbral, labels):
     kmeans = KMeans(data, 7, umbral)
     centroids, clusters = kmeans.fit()

     # Calculate purity
     purity_value = purity(clusters, labels)

     print(f"Purity = {purity_value}")

     # Save purity and details to kmeans_results
     results_kmeans.append([k, umbral, purity_value])

     # Evaluate cluster quality
     cluster_labels = np.unique(clusters)
     for label in cluster_labels:
         cluster_indices = np.where(clusters == label)[0]
         cluster_emotions = labels[cluster_indices]
         unique_emotions, emotion_counts = np.unique(cluster_emotions, return_counts=True)
         most_frequent_emotion = unique_emotions[np.argmax(emotion_counts)]
         print(f"Cluster {label}: Most frequent emotion = {most_frequent_emotion}")

     # Plot the results
     plt.figure()
     plt.scatter(data[:, 0], data[:, 1], c=clusters)
     plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
     plt.savefig(f"K_means_{k}_Umbral_{umbral}.png")

def execute_dbscan(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.dbscan(data)
    purity_value = purity(clusters, labels)
    print(f"Purity = {purity_value}")
    results_dbscan.append([eps, min_samples, purity_value])

    # Evaluate cluster quality
    cluster_labels = np.unique(clusters)
    for label in cluster_labels:
        cluster_indices = np.where(clusters == label)[0]
        cluster_emotions = labels[cluster_indices]
        unique_emotions, emotion_counts = np.unique(cluster_emotions, return_counts=True)
        most_frequent_emotion = unique_emotions[np.argmax(emotion_counts)]
        print(f"Cluster {label}: Most frequent emotion = {most_frequent_emotion}")
    
    # Plot the results
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.savefig(f"dbscan_eps_{eps}_min_samples_{min_samples}.png")

def execute_gmm(data, iterations):
    gmm = GMM(n_components=7, n_iter=iterations)
    gmm.fit(data)
    clusters = np.array(gmm.predict(data))
    purity_value = purity(clusters, labels)
    print(f"Purity = {purity_value}")
    results_gmm.append([iterations, purity_value])

    # Evaluate cluster quality
    cluster_labels = np.unique(clusters)
    for label in cluster_labels:
        cluster_indices = np.where(clusters == label)[0]
        cluster_emotions = labels[cluster_indices]
        unique_emotions, emotion_counts = np.unique(cluster_emotions, return_counts=True)
        most_frequent_emotion = unique_emotions[np.argmax(emotion_counts)]
        print(f"Cluster {label}: Most frequent emotion = {most_frequent_emotion}")
    
    # Plot the results
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=gmm.cluster(data))
    plt.savefig(f"gmm_iterations_{iterations}.png")

if __name__ == "__main__":
    # Set 
    path_file = "./data"
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    
    # Load data
    data_scaled, labels = load_data()

    # # Apply KMeans
    # umbral = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    # dimensions = [2,3,4,5,6,7]
    # for k in dimensions:
    #     for u in umbral:
    #         # Apply PCA
    #         pca = PCA(n_components=k)
    #         data_pca = pca.fit_transform(data_scaled)
    #         execute_kmeans(data_scaled, k, u, labels) 

    # # Save kmeans_results to csv
    # df = pd.DataFrame(results_kmeans, columns=['K', 'Umbral', 'Purity'])
    # df.to_csv('kmeans_results.csv', index=False)

    eps = [0.3]
    min_samples = [3]

    for e in eps:
        for m in min_samples:
            execute_dbscan(data_scaled, e, m)

    # Save dbscan_results to csv
    df = pd.DataFrame(results_dbscan, columns=['Eps', 'Min_samples', 'Purity'])
    df.to_csv('dbscan_results.csv', index=False)
    
    # Save gmm_results to csv
    ite = [2,3,5,7,10,20]
    for i in ite:
        execute_gmm(data_pca,i)
    
    df = pd.DataFrame(results_gmm, columns = ['Iterations', 'Purity'])
    df.to_csv('gmm_results.csv', index=False)
    
