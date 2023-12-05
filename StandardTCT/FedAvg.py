import torch
import torch.nn.functional as F


def client_update(client_model, optimizer, train_loader, epoch=5): #réalise un pas d'optimization et retourne la loss
    """Train a client_model on the train_loder data."""
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data, target #supprimer data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


def average_models(global_model, client_models):
    """Average models across all clients."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0) #stack les résultas et fait la moyenne par client
    global_model.load_state_dict(global_dict)
    return 