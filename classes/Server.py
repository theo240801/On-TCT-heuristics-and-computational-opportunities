from typing import List
from classes.Client import Client
from classes.Algos import Optimizer, Federated


class Server:
    Federate : Optimizer
    Clients: List[Client]
    global_n_epochs:int #le nombre d'apprentissages fédérés
    local_n_epochs:int #le nombre d'itérations locales du client entre chaque apprentissage fédéré

    def train(self):
        """Entraine le serveur avec la descente de gradient de Opti sur les clients de la liste Clients"""
        for i in range(self.global_n_epochs):
            for client in self.Clients:
                (client.Model.weights).Server.Optimizer.update(client.Model)
                

    def FedAvg(..)

    def Scaffold(..)
    
    def TCT(...)