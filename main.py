from __future__ import print_function

from knn.core import KNearestNeighboursCore
import torch

classifier = KNearestNeighboursCore()

classifier.test()
x = torch.rand(5, 3)
print(x)
