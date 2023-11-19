import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, loss_fn_name):
        super(Model, self).__init__()

        # Vérifier que le nombre de couches cachées est au moins 1
        assert len(hidden_sizes) >= 1, "Il doit y avoir au moins une couche cachée."

        # Liste pour stocker les couches cachées
        self.hidden_layers = nn.ModuleList()

        # Création des couches cachées
        for i in range(len(hidden_sizes)):
            if i == 0:
                # Première couche cachée
                self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                # Couches cachées subséquentes
                self.hidden_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        # Couche de sortie
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        # Fonction d'activation ReLU
        self.relu = nn.ReLU()

        # Fonction de perte
        if loss_fn_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_fn_name == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError("Fonction de perte non reconnue.")

        # Optimiseur (vous pouvez ajuster les hyperparamètres si nécessaire)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        # Propagation avant du modèle
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x

    def compute_loss(self, output, target):
        # Calcul de la perte en utilisant la fonction de perte définie
        loss = self.loss_fn(output, target)
        return loss

    def update_weights(self, gradient):
        # Mise à jour des poids du modèle avec le gradient donné
        self.optimizer.zero_grad()
        gradient.backward()
        self.optimizer.step()
    

