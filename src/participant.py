import torch
import torch.nn as nn

import customModules as cm
import parameters as pm

class Participant():
    def __init__(self, participantID, model, server, paramsNum, theta_u, dataloader):
        self.participantID = participantID
        self.module = model()
        self.params = None
        self.paramsNumToUpload = int(paramsNum * theta_u)
        self.server = server
        self.dataloader = dataloader

    def download_params(self):
        self.params = self.server.globalParams
        print(self.participantID + " downloaded_parameters from the server")

    def train_upload(self, n_epochs, DSSGD=False, fedAvg=False):
        print(self.participantID + " training")
        self.module.load_state_dict(self.params.params)
        for epoch in range(n_epochs):
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

            print("Step: {}/{} | Acc:{:.3f}%".format(batch_idx + 1, len(self.dataloader),
                                                                                  100. * correct / total))


        if DSSGD:
            grads = pm.Gradients(self.module.state_dict(), self.params.params)
            top_grads = grads.topN(self.paramsNumToUpload)
            self.server.upload(top_grads)
            print(self.participantID + " uploaded " + str(self.paramsNumToUpload) + "parameters to the server")

        elif fedAvg:
            self.server.upload(self.module.state_dict())