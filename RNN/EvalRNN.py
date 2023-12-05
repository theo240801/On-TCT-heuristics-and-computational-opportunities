import torch
import numpy as np
import torch.nn.functional as F

from StandardTCT.RNNfunctions import evaluate, categoryFromOutput

def evaluate_model(model, data_loader):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data, target #supprimer .cuda()
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc

def evaluate_model_RNN(model, test):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    correct = 0
    for line in test :
        output = evaluate(model, line[0])
        guess, guess_i = categoryFromOutput(output)
        if guess_i == line[1][0].item():
            correct += 1

    acc = correct / len(test)

    return acc, correct

def evaluate_many_models(models, data_loader):
    """Compute average loss and accuracy of multiple models on a data_loader."""
    num_nodes = len(models)
    losses = np.zeros(num_nodes)
    accuracies = np.zeros(num_nodes)
    for i in range(num_nodes):
        losses[i], accuracies[i] = evaluate_model(models[i], data_loader)
    return losses, accuracies

def evaluate_many_models_RNN(models, line_tensors, category_tensor):
    """Compute average loss and accuracy of multiple models on a data_loader."""
    num_nodes = len(models)
    losses = np.zeros(num_nodes)
    accuracies = np.zeros(num_nodes)
    for i in range(num_nodes):
        losses[i], accuracies[i] = evaluate_model_RNN(models[i], line_tensors, category_tensor)
    return losses, accuracies