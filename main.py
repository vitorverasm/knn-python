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
from sklearn.datasets import load_digits
import numpy as np
from knn.core import KNearestNeighborsCore

# Load dataset
digits = load_digits(n_class=10, return_X_y=False)
X = digits.data
Y = digits.target

# Split dataset between train and test data
n_train, n_test = (1000, 100)
x_train, y_train = X[:n_train, :], Y[:n_train]
x_test, y_test = X[n_train:n_train + n_test, :], Y[n_train:n_train + n_test]

# Instance of knn model
knn_instance = KNearestNeighborsCore(x_train, y_train, k=5)

# Params
times = 100
acc = 0

for i in range(times):
    print("\n###########################")
    print("Round: ", i + 1)
    # Choose new test sample
    i_test = np.random.choice(np.arange(x_test.shape[0]))
    xi_test = x_test[i_test]
    yi_test = y_test[i_test]
    yi_predicted = knn_instance.predict(xi_test)
    hit = yi_test == yi_predicted

    # Print output
    # print("xi (test):\n", xi_test)
    print("y:", yi_test, "\ny(predicted):", yi_predicted, "\n(HIT)" if hit else "\n(MISS)")

    # Accuracy
    if hit:
        acc += 1
    print("###########################")

print("\nRESULTS: ")
print("Accuracy = ", (acc / times) * 100, "%")
