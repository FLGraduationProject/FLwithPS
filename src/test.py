import argparse

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import numpy as np

import customModules as cm
import parameterServer as ps
import participant as pt
import dataLoader as dl
import test as test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps_single_interval', type=int, default=10, help='')
    parser.add_argument('--n_participants', type=int, default=90, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--theta_u', type=float, default=0.1, help='')
    parser.add_argument('--local_data_ratio', type=float, default=0.01, help='')
    parser.add_argument('--n_server_update', type=int, default=5, help='')
    parser.add_argument('--n_participants_single_interval', type=int, default=10, help='')
    parser.add_argument('--interval', type=int, default=10, help='')

    '''
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        useGPU = True
        print('학습을 진행하는 기기:', device)
        print('cuda index:', torch.cuda.current_device())
        print('gpu 개수:', torch.cuda.device_count())
        print('graphic name:', torch.cuda.get_device_name())
    else:
        device = None
        useGPU = False
        print('학습을 진행하는 기기: CPU')
        '''

    opt = parser.parse_args()

    train_loader = dl.divideData2Participants(opt.local_data_ratio, opt.n_participants, opt.batch_size)

    initialmodel = cm.SimpleCNN()

    initial_params = initialmodel.state_dict()
    paramsNum = sum(p.numel() for p in initialmodel.parameters() if p.requires_grad)

    server = ps.parameterServer(initial_params)

    participants = []

    for i in range(opt.n_participants):
        participants.append(pt.Participant('device-' + str(i), cm.SimpleCNN, server, paramsNum, opt.theta_u, train_loader[i]))

    for _ in range(opt.interval):
        participants_ready = [False for _ in range(opt.n_participants)]
        participants_this_interval = np.random.permutation(np.arange(opt.n_participants))[:opt.n_participants_single_interval]

        server_update = sorted(np.random.permutation(np.arange(opt.n_steps_single_interval))[:opt.n_server_update - 1])
        server_update_num = 0

        for step in range(opt.n_steps_single_interval):
            participant_num = participants_this_interval[np.random.randint(0, opt.n_participants_single_interval)]
            if participants_ready[participant_num]:
                participants[participant_num].train_upload()
                participants_ready[participant_num] = False

            else:
                participants[participant_num].download_params()
                participants_ready[participant_num] = True

            if server_update_num == opt.n_server_update - 1:
                continue
            if server_update[server_update_num] == step:
                server.update()
                server_update_num += 1
        server.update()
        print('---------------------------------------------------interval over')

        # Test global model
        test.test(opt.batch_size, server.globalParams.params)