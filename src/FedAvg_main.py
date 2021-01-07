import argparse

import torch.nn as nn
import numpy as np

import customModules as cm
import parameterServer as ps
import participant as pt
import dataLoader as dl
import test as test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_participants', type=int, default=90, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--theta_u', type=float, default=0.1, help='')
    parser.add_argument('--local_data_ratio', type=float, default=0.01, help='data size ratio each participant has')
    parser.add_argument('--n_participants_single_round', type=int, default=6, help='')
    parser.add_argument('--n_rounds', type=int, default=200, help='')
    parser.add_argument('--model_type', type=nn.Module, default=cm.SimpleDNN)
    parser.add_argument('--n_local_epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.01)
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

    initialmodel = opt.model_type()

    initial_params = initialmodel.state_dict()
    paramsNum = sum(p.numel() for p in initialmodel.parameters() if p.requires_grad)

    server = ps.parameterServer(initial_params)

    participants = []

    for i in range(opt.n_participants):
        participants.append(pt.Participant('device-' + str(i), opt.model_type, server, paramsNum, opt.theta_u, train_loader[i]))

    for _ in range(opt.n_rounds):
        participants_this_round = np.random.permutation(np.arange(opt.n_participants))[:opt.n_participants_single_round]

        for participant_num in participants_this_round:
            participants[participant_num].download_params()
            participants[participant_num].train_upload(opt.n_local_epochs, fedAvg=True)

        server.update(fedAvg=True)

        print('---------------------------------------------------round over')

        # Test global model
        test.test(opt.model_type, opt.batch_size, server.globalParams.params)