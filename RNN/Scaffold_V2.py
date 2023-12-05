from RNN.Utils import *
import torch.nn.functional as F

def client_update_Scaffold_V2(model, client_model, optimizer, train_loader, c, c_i, epoch=5):
        """Local training"""

        client_model.train()
        # client_model.to(device)
        loss = 0
        for _ in range(epoch):
            for data, targets in train_loader:
                loss += scaffold_step(client_model, optimizer, data, targets, c, c_i)

        # update control variates for scaffold algorithm
        c_i_plus, c_update_amount = update_control_variate(model, client_model, c, c_i)

        return loss, c_i_plus, c_update_amount

def scaffold_step(model, optimizer, data, targets, c, c_i):
    optimizer.zero_grad()

    # forward pass
    # data, targets = data.to(self.device), targets.to(self.device)
    logits = model(data)
    loss = F.nll_loss(logits, targets)

    # backward pass
    loss.backward()
    grad_batch = flatten_grads(model).detach().clone()
    optimizer.zero_grad()

    # add control variate
    grad_batch = grad_batch - c_i + c
    model = assign_grads(model, grad_batch)
    optimizer.step()

    return loss.item()

def update_control_variate(model, client_model, c, c_i, lr=0.1, n_epoch_local=5):

    divisor = 1.0/(n_epoch_local * lr)

    server_params = flatten_weights(client_model)
    local_params = flatten_weights(model)
    param_move = server_params - local_params

    c_i_plus = c_i - c + (divisor * param_move)
    c_update_amount = c_i_plus - c_i

    return c_i_plus, c_update_amount