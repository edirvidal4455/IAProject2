class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.visited = set()

    def dbscan(self, dataset):
        self.labels = [0] * len(dataset)
        cluster_label = 0

        for i, point in enumerate(dataset):
            if i in self.visited:
                continue
            self.visited.add(i)
            neighbors = self._region_query(dataset, point)

            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                cluster_label += 1
                self._expand_cluster(dataset, i, neighbors, cluster_label)

        return self.labels

    def _expand_cluster(self, dataset, point_index, neighbors, cluster_label):
        self.labels[point_index] = cluster_label

        while neighbors:
            current_point = neighbors.pop(0)
            if current_point not in self.visited:
                self.visited.add(current_point)
                current_neighbors = self._region_query(dataset, dataset[current_point])

                if len(current_neighbors) >= self.min_samples:
                    neighbors.extend(current_neighbors)

            if self.labels[current_point] == 0:
                self.labels[current_point] = cluster_label

    def _region_query(self, dataset, point):
        neighbors = []
        for i, neighbor in enumerate(dataset):
            if self._euclidean_distance(point, neighbor) <= self.eps:
                neighbors.append(i)
        return neighbors

    @staticmethod
    def _euclidean_distance(point_a, point_b):
        squared_distance = sum([(a - b) ** 2 for a, b in zip(point_a, point_b)])
        return squared_distance ** 0.5


# Example usage
if __name__ == '__main__':
    dataset = [[1, 1], [1, 2], [2, 1], [10, 10], [10, 11], [11, 10], [20, 20]]
    dbscan = DBSCAN(eps=3, min_samples=2)
    labels = dbscan.dbscan(dataset)
    print(labels)