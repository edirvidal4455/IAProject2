class DBScan:

    def __init__(self, epsilon, puntos_minimos): #parametro de entrada
        self.puntos_minimos =puntos_minimos
        self.eps = epsilon
        self.labels = None
        self.visited = set()

    def fit(self, data):
        self.labels=np.zeros(len(data))   # Un array  de ceros con len de X
        cluster_label = 0                  #esto nos ira por cada punto 

        for indice, punto in enumerate(data):
            if indice in self.visited:
                continue
            self.visited.add(indice)
            neighbors = self.vecinos_cercanos(data, punto)

            if len(neighbors) < self.puntos_minimos:
                self.labels[indice] = -1
            else:
                cluster_label =cluster_label + 1
                self.explandir_cluster(data, indice, neighbors, cluster_label)

        return self.labels

    def explandir_cluster(self, dataset, point_index, vecinos, cluster_label):

        self.labels[point_index] = cluster_label

        while vecinos:
            current_point = vecinos.pop(0)
            if current_point not in self.visited:
                self.visited.add(current_point)
                current_neighbors = self.vecinos_cercanos(dataset, dataset[current_point])

                if len(current_neighbors) >= self.puntos_minimos:
                    vecinos.extend(current_neighbors)

            if self.labels[current_point] == 0:
                self.labels[current_point] = cluster_label

    def vecinos_cercanos(self, datafrane, punto): # agregamos las los vecinos que estan dentro eps
        vecinos = []
        for i, vecino in enumerate(datafrane):
            if self.distancia(punto, vecino) <= self.eps:
                vecinos.append(i)
        return vecinos

    def distancia(a, b): #calculamos la distancia con euclidiana
        dis = (sum([(a - b) ** 2 for a, b in zip(a,b)]))**0.5
        return dis
