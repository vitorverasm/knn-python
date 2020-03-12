import torch


def reader(path):
    samples, target = [], []
    with open(path, 'r') as csv_file:
        for line in csv_file.readlines():
            line_ = list(map(float, line.split(',')))
            samples.append(line_[:2])
            target.append(line_[2])
    return torch.tensor(samples), torch.tensor(target)
