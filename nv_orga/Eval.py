import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def evaluate_model(model, data_loader):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    loss = 0
    correct = 0

    criterion = nn.CrossEntropyLoss()


    with torch.no_grad():
        for data, target in data_loader:
            data, target = data, target #supprimer .cuda()
            output = model(data)
            #loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc

def evaluate_many_models(models, data_loader):
    """Compute average loss and accuracy of multiple models on a data_loader."""
    num_nodes = len(models)
    losses = np.zeros(num_nodes)
    accuracies = np.zeros(num_nodes)
    for i in range(num_nodes):
        losses[i], accuracies[i] = evaluate_model(models[i], data_loader)
    return losses, accuracies