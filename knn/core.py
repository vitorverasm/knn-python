import numpy as np
from scipy import stats


class KNearestNeighborsCore:
    def __init__(self, x_train, y_train, k=3):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    @staticmethod
    def euclidean_distance(a, b):
        return np.linalg.norm(a - b)

    def predict(self, test_sample):
        distances = np.zeros((self.x_train.shape[0],))
        for index, x_i in enumerate(self.x_train):
            distances[index] = self.euclidean_distance(x_i, test_sample)
        neighbours_x = distances.argsort()[:self.k]
        predicted_y = stats.mode(self.y_train[neighbours_x]).mode[0]
        return predicted_y


def execution(self):
    """
    1. Load the data
    2. Initialise the value of k
    3. For getting the predicted class, iterate from 1 to total number of training data points
        3.1. Calculate the distance between test data and each row of training data. Here we will use Euclidean
        distance as our distance metric since it’s the most popular method. The other metrics that can be used are
        Chebyshev, cosine, etc.
        3.2. Sort the calculated distances in ascending order based on distance values
        3.3. Get top k rows from the sorted array
        3.4. Get the most frequent class of these rows
        3.5. Return the predicted class
    """
    pass
