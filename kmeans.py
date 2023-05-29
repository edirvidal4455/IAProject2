# Imprt libraries
import numpy as np

"""
KMeans
"""
def distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def Init_Centroide(data, k):
    # Obtener valores random
    random_values = np.random.choice(data.shape[0], k, replace=False)
    # Retornar los centroides
    return data[random_values]

def return_new_centroide(grupos, data, k):
    # Initialize centroids
    new_centroide = np.zeros((k, data.shape[1]))

    for i in range(k):
        # Get all points assigned to a specific cluster
        new_centroide[i] = np.mean(data[grupos == i], axis=0)

    return new_centroide

def get_cluster(data, centroides):

    # Initialize array
    grupos = np.zeros(data.shape[0], dtype=np.int64)
    
    for i in range(data.shape[0]):
        # Initialize distances vector for each centroid
        distancias = np.zeros(centroides.shape[0])

        for j in range(centroides.shape[0]):
            # Calculate distance between point and centroid
            distancias[j] = distance(data[i], centroides[j])
        # Asign cluster to point
        grupos[i] = np.argmin(distancias)

    return grupos

def distancia_promedio_centroides(old_centroides, new_centroides):
    # Initialize array of distances
    promedios = []
    # Iterate each
    for i in range(old_centroides.shape[0]):
        # Get distance between them
        dist = distance(old_centroides[i], new_centroides[i])
        # Append to mean
        promedios.append(dist)
    # Return mean value
    return np.mean(promedios)

# Este es el algoritmo K-Means. Debe retornar los centroides y los clusters 
# generados para poder utilizarlos en análisis posteriores. 
def kmeans_algo(data, k, umbral):
  centroides =  Init_Centroide(data,k)
  clusters   =  get_cluster(data,centroides)
  new_centroides = return_new_centroide(clusters, data,k)
  while(distancia_promedio_centroides(centroides, new_centroides) > umbral):
     centroides = new_centroides
     clusters   =  get_cluster(data,centroides)
     new_centroides = return_new_centroide(clusters, data,k)
  return new_centroides, clusters


class KMeans:
    def __init__(self, data, k, umbral):
        self.data = data
        self.k = k
        self.umbral = umbral

    def fit(self):
        return kmeans_algo(self.data, self.k, self.umbral)