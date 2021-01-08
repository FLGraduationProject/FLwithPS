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
    parser.add_argument('--local_data_ratio', type=float, default=0.01, help='data size ratio each participant has')
    parser.add_argument('--n_server_update_single_interval', type=int, default=5, help='')
    parser.add_argument('--n_participants_single_interval', type=int, default=6, help='')
    parser.add_argument('--n_intervals', type=int, default=200, help='')
    parser.add_argument('--model_type', type=nn.Module, default=cm.SimpleCNN)
    parser.add_argument('--n_local_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    # 개념적으로는 server_update가 동일한 시간마다 일어난다고 가정
    # interval은 lagging에 한계를 주기 위해서 maximum lagging이 일어날 수 있는 시간 간격
    # steps single interval이 많을 수록 interval 내에서 download와 upload 횟수가 늘어난다
    # n_participants_single_interval이 클 수록 interval 별로 participant 수가 적어져서 일정 수준의 upload를 보장한다.
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

    train_loader = dl.divideData2Participants(opt.local_data_ratio, opt.n_participants, opt.batch_size, eq_IID=True)

    initialmodel = opt.model_type()

    initial_params = initialmodel.state_dict()
    paramsNum = sum(p.numel() for p in initialmodel.parameters() if p.requires_grad)

    server = ps.parameterServer(initial_params)

    participants = []

    for i in range(opt.n_participants):
        participants.append(pt.Participant('device-' + str(i), opt.model_type, server, paramsNum, opt.theta_u, train_loader[i]))

    for _ in range(opt.n_intervals):
        participants_ready = [False for _ in range(opt.n_participants)]
        participants_this_interval = np.random.permutation(np.arange(opt.n_participants))[:opt.n_participants_single_interval]

        server_update = sorted(np.random.permutation(np.arange(opt.n_steps_single_interval))[:opt.n_server_update_single_interval - 1])
        server_update_num = 0

        n_upload = 0

        for step in range(opt.n_steps_single_interval):
            participant_num = participants_this_interval[np.random.randint(0, opt.n_participants_single_interval)]
            if participants_ready[participant_num]:
                n_upload += 1
                participants[participant_num].train_upload(opt.n_local_epochs, DSSGD=True)
                participants_ready[participant_num] = False

            else:
                participants[participant_num].download_params()
                participants_ready[participant_num] = True

            if server_update_num == opt.n_server_update_single_interval - 1:
                continue
            if server_update[server_update_num] == step:
                server.update(DSSGD=True)
                server_update_num += 1
        server.update()
        print(str(n_upload) + ' uploaded in this interval')
        print('---------------------------------------------------interval over')

        # Test global model
        test.test(opt.model_type, opt.batch_size, server.globalParams.params)