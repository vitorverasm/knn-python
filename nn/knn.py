import torch


# torch.arange
# torch.linespace

class KNN(object):
    def __init__(self, K, samples_train, samples_targets):
        self.K = K
        self.samples = samples_train
        self.targets = samples_targets

    def predict(self, x):
        cdists = torch.cdist(x, self.samples)
        idxs = cdists.argsort()[:, :self.K]
        knns = self.targets[idxs]

        return knns.mode(dim=1).values
