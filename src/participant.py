import torch
import torch.nn as nn

import customModules as cm
import parameters as pm

class Participant():
    def __init__(self, deviceID, server, paramsNum, theta_u, dataloader):
        self.deviceID = deviceID
        self.module = cm.SimpleCNN()
        self.params = None
        self.paramsNumToUpload = int(paramsNum * theta_u)
        self.server = server
        self.dataloader = dataloader

    def download_params(self):
        self.params = self.server.globalParams
        print(self.deviceID + " downloaded_parameters from the server")

    def train_upload(self):
        print(self.deviceID + " training")
        self.module.load_state_dict(self.params.params)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.module.parameters(), lr=0.01)

        self.module.train()
        train_loss = 0
        total = 0
        correct = 0

        for batch_idx, data in enumerate(self.dataloader):
            image, label = data
            # Grad initialization
            optimizer.zero_grad()
            # Forward propagation
            output = self.module(image)
            # Calculate loss
            loss = criterion(output, label)
            # Backprop
            loss.backward()
            # Weight update
            optimizer.step()

            train_loss += loss.item()

            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if (batch_idx + 1) % 1000 == 0:
                print("Step: {}/{} | train_loss: {:.4f} | Acc:{:.3f}%".format(batch_idx + 1, len(self.dataloader),
                                                                              train_loss / 1000,
                                                                              100. * correct / total))

        grads = pm.Gradients(self.module.state_dict(), self.params.params)
        top_grads = grads.topN(self.paramsNumToUpload)
        self.server.upload(top_grads)
        print(self.deviceID + " uploaded " + str(self.paramsNumToUpload) + "parameters to the server")