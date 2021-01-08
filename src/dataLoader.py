import torch
import torchvision
from torchvision import transforms

import numpy as np


def divideData2Participants(local_data_ratio, n_participants, batch_size, eq_IID=False, dif_IID=False, Non_IID=False):
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    if eq_IID:
        traindata_split = torch.utils.data.random_split(trainset, [int(trainset.data.shape[0] * local_data_ratio) for _ in
                                                                   range(int(1/local_data_ratio))])
        train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]


    elif dif_IID:
        traindata_split = torch.utils.data.random_split(trainset, np.random.multinomial(trainset.data.shape[0]-n_participants, np.ones(n_participants)/n_participants)+1)
        train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]


    elif Non_IID:
        class_inds = [torch.where(trainset.targets == class_idx)[0]
                      for class_idx in trainset.class_to_idx.values()]
        sorted_inds = torch.cat(class_inds)
        each_parti_data_size = (np.random.multinomial(len(sorted_inds)-n_participants, np.ones(n_participants)/n_participants)+1).tolist()
        each_parti_inds = torch.split(sorted_inds, each_parti_data_size)

        train_loader = [
            torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, inds), batch_size=1, shuffle=False) for inds
            in each_parti_inds]


    return train_loader