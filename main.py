import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pywt
from kmeans import KMeans
from gmm import GMM
from dbscan import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

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
        cluster_labels = labels[cluster_indices]
        unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
        total += np.max(label_counts)
    return total / len(labels)

def execute_kmeans(data, k, umbral, labels):
     kmeans = KMeans(data, k, umbral)
     centroids, clusters = kmeans.fit()

     # Calculate purity
     purity_value = purity(clusters, labels)

     print(f"Purity = {purity_value}")

     # Save purity to file
     with open(f"purity_K_{k}_Umbral_{umbral}.txt", "w") as f:
         f.write(str(purity_value))

     # Evaluate cluster quality
     cluster_labels = np.unique(clusters)
     for label in cluster_labels:
         cluster_indices = np.where(clusters == label)[0]
         cluster_emotions = labels[cluster_indices]
         unique_emotions, emotion_counts = np.unique(cluster_emotions, return_counts=True)
         most_frequent_emotion = unique_emotions[np.argmax(emotion_counts)]
         print(f"Cluster {label}: Most frequent emotion = {most_frequent_emotion}")

     # Plot the results
     plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    #  plt.show()
     plt.savefig(f"plot_K_{k}_Umbral_{umbral}.png")

if __name__ == "__main__":
    # Set 
    path_file = "./data"
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    
    # Load data
    data_scaled, labels = load_data()

    # Apply PCA
    pca = PCA(n_components=7)
    
    data_pca = pca.fit_transform(data_scaled)
    execute_kmeans(data_pca, 7, 0.0001, labels)
    
    # Apply KMeans

    umbral = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    dimensions = [1,2,3,4,5,6,7]
    
   