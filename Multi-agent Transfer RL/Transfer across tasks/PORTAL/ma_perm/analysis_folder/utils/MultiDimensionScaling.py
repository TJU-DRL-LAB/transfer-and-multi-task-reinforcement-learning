import numpy as np
from scipy.spatial.distance import cdist


class MDS():
    def __init__(self):
        self.eps = 1e-5

    def construct_gram_matrix(self, distance_matrix):
        # M_ij = (D_1j^2 + D_i1^2 - D_ij^2) / 2
        row, col = distance_matrix.shape
        gram = np.zeros([row, col])
        for i in range(row):
            for j in range(col):
                gram[i][j] = 0.5 * ((distance_matrix[0][j])**2 + (distance_matrix[i][0])**2 - (distance_matrix[i][j])**2)
        return gram

    def get_coordinates_by_distance(self, distance_matrix):
        gram_matrix = self.construct_gram_matrix(distance_matrix)
        # get eigenvalue and eigenvectors
        feature_values, feature_vectors = np.linalg.eig(gram_matrix)
        select_feature_values = []

        # if eigenvalue == 0, delete it and its eigenvector
        for i in range(len(feature_values) - 1, -1, -1):
            if feature_values[i] > self.eps:
                select_feature_values.append(feature_values[i])
            else:
                feature_vectors = np.delete(feature_vectors, i, axis=1)
        eye_matrix = np.eye(len(select_feature_values))
        select_feature_values.reverse()
        # sqrt(eigenvalues) * eigenvectors
        for i in range(len(select_feature_values)):
            eye_matrix[i, i] = select_feature_values[i]
        return np.dot(feature_vectors, eye_matrix**0.5)


if __name__ == '__main__':
    points = np.array([[0, 0], [1, 0], [2, 1], [1,2]])
    mds = MDS()
    origin_dist_matrix = cdist(points, points, 'euclidean')
    coordinates = mds.get_coordinates_by_distance(origin_dist_matrix)
    new_dist_matrix = cdist(coordinates, coordinates, 'euclidean')
    print(origin_dist_matrix)
    print(new_dist_matrix)

    exit(0)
