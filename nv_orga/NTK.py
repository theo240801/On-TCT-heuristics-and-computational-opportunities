import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def compute_eNTK(model, X, subsample_size=100000, seed=123):
    """"compute eNTK
    
    prend un modèle de la classe Net_eNTK et un tenseur X en entrée et renvoie une matrice de taille (X.size()[0], subsample_size) contenant les vecteurs de gradient subsamplés pour chaque échantillon de X.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model.eval()

    params = list(model.parameters()) #liste de tous les paramètres trainable du modèle
    n_params = sum(p.numel() for p in list(model.parameters()))
    random_index = torch.randperm(n_params)[:subsample_size]

    #params = list(model.parameters())[:-2]  # Exclude the last two layers (self.fc2)
    #n_params = sum(p.numel() for p in list(model.parameters())[:-2])
    #random_index = torch.randperm(n_params)[:subsample_size]

    grads = None
    for i in range(X.size()[0]):
        model.zero_grad() #réinitialise les gradients
        model.forward(X[i : i+1])[0].backward() #calcule les gradients  

        grad = []
        for param in params: #param.requires_grad dans PyTorch est un attribut des tenseurs (y compris ceux qui représentent les paramètres d'un modèle, comme les poids et les biais d'un réseau de neurones) qui détermine si PyTorch doit calculer les gradients pour ces tenseurs pendant la rétropropagation.
            if param.requires_grad:
                #print(param.grad.shape, (np.array(param.grad) == 0).sum())
                grad.append(param.grad.flatten())
        grad = torch.cat(grad) #Concatène tous les gradients aplatis en un seul vecteur
        print(grad.shape, (np.array(grad) == 0).sum())
        grad = grad[random_index] #Réduit la dimensionnalité du vecteur de gradient en utilisant l'indexation aléatoire définie par random_index

        if grads is None:
            #grads = torch.zeros((X.size()[0], grad.size()[0]), dtype=torch.half)
            grads = torch.zeros((X.size()[0], grad.size()[0]))
            #Si grads est None, une matrice zéro de la forme appropriée est initialisée.
        grads[i, :] = grad #Stocke le vecteur de gradient subsamplé pour l'échantillon i dans la matrice grads

    return grads

def client_compute_eNTK(client_model, train_loader):
    """Train a client_model on the train_loder data."""
    client_model.train()

    grads_data = []
    targets_onehot = []
    targets = []

    for data, target in train_loader:
        grads = compute_eNTK(client_model, data)
        grads = grads.float()
        target_onehot = F.one_hot(target, num_classes=10) - (1.0 / 10.0)

        grads_data.append(grads)
        targets_onehot.append(target_onehot)
        targets.append(target)

    grads_data = torch.cat(grads_data)
    targets_onehot = torch.cat(targets_onehot)
    targets = torch.cat(targets)

    return grads_data, targets_onehot, targets