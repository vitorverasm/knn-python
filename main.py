from sklearn.datasets import load_digits
import numpy as np
from knn.core import KNearestNeighborsCore
from knn.utils import Utils

# Load dataset
digits = load_digits(n_class=10, return_X_y=False)
splited_dataset = Utils.split_dataset(digits)

# Instance of knn model
knn_instance = KNearestNeighborsCore(splited_dataset["x_train"], splited_dataset["y_train"], k=5)

# Params
times = 100
acc = 0

for i in range(times):
    print("\n###########################")
    print("Round: ", i + 1)
    # Choose new test sample
    i_test = np.random.choice(np.arange(splited_dataset["x_test"].shape[0]))
    xi_test = splited_dataset["x_test"][i_test]
    yi_test = splited_dataset["y_test"][i_test]
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
