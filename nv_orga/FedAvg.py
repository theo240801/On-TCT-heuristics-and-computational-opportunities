import torch
import torch.nn.functional as F
import torch.nn as nn

def client_update(client_model, optimizer, train_loader, epoch=5): #réalise un pas d'optimization et retourne la loss
    """Train a client_model on the train_loder data."""
    client_model.train()
    criterion = nn.CrossEntropyLoss()

    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available() :
                data, target = data.cuda(), target.cuda() # Add .cuda() to use GPU
            #print('target', target)
            optimizer.zero_grad()
            output = client_model(data)
            #loss = F.nll_loss(output, target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


def average_models(global_model, client_models):
    """Average models across all clients."""
    global_dict = global_model.state_dict()
    total_params = 0  # Variable to store the total number of parameters

    for k in global_dict.keys():
        if k.startswith('conv') or k.startswith('layer') or k.startswith('fc'):
            if torch.cuda.is_available() :
                global_dict[k] = torch.stack([client_models[i].state_dict()[k].float().cuda() for i in range(len(client_models))], 0).mean(0) # Add .cuda() to use GPU
            else :
                global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0) # Add .cuda() to use GPU

            layer_params = [client_models[i].state_dict()[k] for i in range(len(client_models))]
            #global_dict[k] = torch.stack(layer_params, 0).mean(0)
            total_params += sum(p.numel() for p in layer_params)  # Count the number of parameters in the layer

    global_model.load_state_dict(global_dict)
    return total_params
