import os
import pywt
import numpy as np
from sklearn.mixture import GaussianMixture
from kmeans import kmeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def perform_pca(features, desired_dimension):
    pca = PCA(n_components=desired_dimension)
    reduced_features = pca.fit_transform(features)
    return reduced_features

def extract_wavelet_features(picture, cortes):
    LL = picture
    for i in range(cortes):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    return LL.flatten()

def calculate_purity(labels_true, labels_pred):
    cm = confusion_matrix(labels_true, labels_pred)
    purity = np.sum(np.max(cm, axis=1)) / np.sum(cm)
    return purity


# Load data
path_file = "./data" 
emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

images = []
labels = []

# Load the images and labels from the folders
for emotion in emotions:
    folder_path = os.path.join(path_file, emotion)
    for file_name in os.listdir(folder_path):
         if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(folder_path, file_name)
            images.append(image_path)
            labels.append(emotion)

extracted_features = []
for image in images:
    image = plt.imread(image)
    features = extract_wavelet_features(image, cortes=3)
    extracted_features.append(features)

extracted_features = np.array(extracted_features)

reduced_features = perform_pca(extracted_features, desired_dimension=2)

kmeans = KMeans(n_clusters=7)
kmeans.fit(reduced_features)
clusters = kmeans.predict(reduced_features)

# Calculate purity
# Change labels to number
labels_number = []
for label in labels:
    if label == 'anger':
        labels_number.append(0)
    elif label == 'contempt':
        labels_number.append(1)
    elif label == 'disgust':
        labels_number.append(2)
    elif label == 'fear':
        labels_number.append(3)
    elif label == 'happy':
        labels_number.append(4)
    elif label == 'sadness':
        labels_number.append(5)
    elif label == 'surprise':
        labels_number.append(6)
print(labels_number)
print(clusters)
purity = calculate_purity(labels_number, clusters)
print("Purity: ", purity)

# Define colors for the clusters
cluster_colors = sns.color_palette('hls', len(np.unique(clusters)))

# Visualize clusters with legend
plt.figure(figsize=(10, 10))
for cluster_id in np.unique(clusters):
    mask = clusters == cluster_id
    plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], c=[cluster_colors[cluster_id]], s=50)

# Plot legend with colors
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[i], markersize=10) for i in np.unique(clusters)]
plt.legend(legend_elements, np.unique(clusters), loc='best')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KMeans Clustering')
plt.show()