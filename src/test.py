import torch

import dataLoader as dl
import customModules as cm


def test(batch_size, parameters):
    print("--------------------Test data---------------------")
    testloader = dl.testLoader(batch_size)

    # This is for test data
    testmodel = cm.SimpleCNN()
    testmodel.load_state_dict(parameters)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            image, label = data
            output = testmodel(image)

            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if (batch_idx + 1) == len(testloader):
                print("Test_Acc:{:.3f}%".format(100. * correct / total))
    print("--------------------------------------------------")