import torch
import torchvision
from torchvision import transforms

def divideData2Participants(local_data_ratio, n_participants, batch_size):
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    traindata_split = torch.utils.data.random_split(trainset, [int(trainset.data.shape[0] * local_data_ratio) for _ in
                                                               range(int(1/local_data_ratio))])

    # Creating a pytorch loader for a Deep Learning model
    train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

    return train_loader


def testLoader(batch_size):
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return testloader