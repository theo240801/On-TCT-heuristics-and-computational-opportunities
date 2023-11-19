def next_fn(server_weights, federated_dataset):
  # Broadcast the server weights to the clients.
  server_weights_at_client = broadcast(server_weights)

  # Each client computes their updated weights.
  client_weights = client_update(federated_dataset, server_weights_at_client)

  # The server averages these updates.
  mean_client_weights = mean(client_weights)

  # The server updates its model.
  server_weights = server_update(mean_client_weights)

  return server_weights

def client_update(model, dataset, server_weights, client_optimizer):
    """Effectue l'entraînement (en utilisant les poids du modèle du serveur) sur l'ensemble de données du client."""
    # Initialiser le modèle du client avec les poids actuels du serveur.
    client_weights = [param.clone() for param in model.parameters()]
    
    # Assigner les poids du serveur au modèle du client.
    for client_param, server_param in zip(client_weights, server_weights):
        client_param.data.copy_(server_param.data)

    # Utiliser le client_optimizer pour mettre à jour le modèle local.
    for batch in dataset:
        # Utiliser le contexte de gradient.
        with torch.autograd.set_grad_enabled(True):
            # Calculer une passe avant sur le lot de données.
            outputs = model(batch)

            # Calculer le gradient correspondant.
            grads = torch.autograd.grad(outputs.loss, client_weights, retain_graph=True)
            grads_and_vars = zip(grads, client_weights)

            # Appliquer le gradient en utilisant un optimiseur client.
            client_optimizer.zero_grad()
            for param, grad in grads_and_vars:
                param.grad = grad
            client_optimizer.step()

    return client_weights
