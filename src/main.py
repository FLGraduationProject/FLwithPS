import argparse

import torch
import torchvision
from torchvision import transforms
import numpy as np

import customModules as cm
import parameterServer as ps
import participant as pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_rounds', type=int, default=5, help='')
    parser.add_argument('--n_participants', type=int, default=20, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--theta_u', type=float, default=0.1, help='')

    opt = parser.parse_args()

    num_of_device_in_round = 2

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    traindata_split = torch.utils.data.random_split(trainset, [int(trainset.data.shape[0] / opt.n_participants) for _ in
                                                               range(opt.n_participants)])

    # Creating a pytorch loader for a Deep Learning model
    train_loader = [torch.utils.data.DataLoader(x, batch_size=opt.batch_size, shuffle=True) for x in traindata_split]

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)



    initialmodel = cm.SimpleCNN()

    initial_params = initialmodel.state_dict()
    paramsNum = sum(p.numel() for p in initialmodel.parameters() if p.requires_grad)

    server = ps.parameterServer(initial_params)

    devices = []
    devices_selected = []

    for i in range(opt.n_participants):
        devices.append(pt.Participant('device-' + str(i), server, paramsNum, opt.theta_u, train_loader[i]))

    for round in range(opt.n_rounds):
        devices_selected.append(np.random.permutation(np.arange(opt.n_participants))[:num_of_device_in_round])

    for round in range(opt.n_rounds):
        for de_num in devices_selected[round]:
            devices[de_num].download_params()
            devices[de_num].train_upload()
        server.update()

    #Test global model



