class Federated:
    """
    Cette classe contient les méthodes de féderation à savoir ce que fait le serveur au moment
    de rassembler les données de tous les clients.
    """
    def __init__(self, nom):
        self.nom = nom
        self.federate = None  
        self.local_opti = None

    def set_local_function(self, train_local_function):
        # Méthode pour définir la fonction train_local
        self.train_local = train_local_function

    def set_federated_learning(self, federate_function):
        # Méthode pour définir la fonction train_local
        self.federate = federate_function