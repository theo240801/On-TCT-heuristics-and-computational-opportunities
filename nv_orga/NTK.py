import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



class Net_eNTK(nn.Module):
    def __init__(self):
        super(Net_eNTK, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def compute_eNTK(model, X, subsample_size=100000, seed=123):
    """"compute eNTK"""
    model.eval()
    params = list(model.parameters()) #liste de tous les paramètres trainable du modèle
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random_index = torch.randperm(355073)[:subsample_size]
    grads = None
    for i in tqdm(range(X.size()[0])):
        model.zero_grad() #réinitialise les gradients
        model.forward(X[i : i+1])[0].backward() #calcule les gradients  

        grad = []
        for param in params: #param.requires_grad dans PyTorch est un attribut des tenseurs (y compris ceux qui représentent les paramètres d'un modèle, comme les poids et les biais d'un réseau de neurones) qui détermine si PyTorch doit calculer les gradients pour ces tenseurs pendant la rétropropagation.
            if param.requires_grad:
                grad.append(param.grad.flatten())
        grad = torch.cat(grad) #Concatène tous les gradients aplatis en un seul vecteur
        grad = grad[random_index] #Réduit la dimensionnalité du vecteur de gradient en utilisant l'indexation aléatoire définie par random_index

        if grads is None:
            grads = torch.zeros((X.size()[0], grad.size()[0]), dtype=torch.half) #Si grads est None, une matrice zéro de la forme appropriée est initialisée.
        grads[i, :] = grad #Stocke le vecteur de gradient subsamplé pour l'échantillon i dans la matrice grads

    return grads

def client_compute_eNTK(client_model, train_loader):
    """Train a client_model on the train_loder data."""
    client_model.train()

    data, targets = next(iter(train_loader))
    grads_data = compute_eNTK(client_model, data) #supprimer .cuda()
    grads_data = grads_data.float() #supprimer .cuda()
    targets = targets #supprimer .cuda()
    # gradient
    targets_onehot = F.one_hot(targets, num_classes=10) - (1.0 / 10.0) #supprimer .cuda()
    del data
    torch.cuda.empty_cache()
    return grads_data, targets_onehot, targets