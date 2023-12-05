import torch
import torch.nn as nn
import torch.nn.functional as F

def scaffold_update(grads_data, targets, theta_client, h_i_client_pre, theta_global,
                    M=200, lr_local=0.00001):
    # set up data / eNTK
    grads_data = grads_data.float() #supprimer .cuda()
    targets = targets #supprimer .cuda()

    # compute transformed onehot label
    targets_onehot = F.one_hot(targets, num_classes=10) - (1.0 / 10.0) #supprimer .cuda()
    num_samples = targets_onehot.shape[0]

    # compute updates
    h_i_client_update = h_i_client_pre + (1 / (M * lr_local)) * (theta_global - theta_client)
    theta_hat_local = (theta_global) * 1.0

    # local gd
    for local_iter in range(M):
        theta_hat_local -= lr_local * ((1.0 / num_samples) * grads_data.t() @ (grads_data @ theta_hat_local - targets_onehot) - h_i_client_update)

    del targets
    del grads_data
    torch.cuda.empty_cache()
    return theta_hat_local, h_i_client_update



# import torch
# from torch.optim import Optimizer

# class CustomOptimizer(Optimizer):
#     def __init__(self, params, learning_rate=0.001, momentum=0.9):
#         # Call the constructor of the base class (Optimizer)
#         super(CustomOptimizer, self).__init__(params, defaults={})

#         # Define hyperparameters
#         self.learning_rate = learning_rate
#         self.momentum = momentum

#     def step(self, h_i, closure=None):
#         # Perform the optimization step for each parameter
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 # Update the parameter using the gradient and hyperparameters
#                 d_p = p.grad.data
#                 p.data.add_(-self.learning_rate, d_p)
                
#                 # Add the Scaffold update
#                 p.data.add_(self.learning_rate, h_i)

#         return loss.item()  # Return the loss value (optional)

# def client_update_Scaffold(client_model, optimizer, train_loader, h_i_client_pre, epoch=5):
#     # compute updates
#     h_i_client_update = h_i_client_pre + (1 / (M * lr_local)) * (theta_global - theta_client)

#  """Train a client_model on the train_loder data."""
#     client_model.train()
#     for e in range(epoch):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data, target #supprimer data.cuda(), target.cuda()
#             optimizer.zero_grad()
#             output = client_model(data)
#             loss = F.nll_loss(output, target)
#             loss.backward()
#             optimizer.step(h_i_client_update)
#     return loss.item(), h_i_client_update

# def client_update_Scaffold_V2(client_model, optimizer, train_loader, h_i_client_pre, epoch=5):
#         """Local training"""

#         # Keep global model weights
#         self._keep_global()

#         self.model.train()
#         self.model.to(self.device)

#         local_size = self.datasize

#         for _ in range(epochs):
#             for data, targets in self.trainloader:
#                 self._scaffold_step(data, targets)

#         # update control variates for scaffold algorithm
#         c_i_plus, c_update_amount = self._update_control_variate()

#         local_results = self._get_local_stats()

#         return local_results, local_size, c_i_plus, c_update_amount

#     def download_global(self, server_weights, server_optimizer, c, c_i):
#         """Load model & Optimizer"""
#         self.model.load_state_dict(server_weights)
#         self.optimizer.load_state_dict(server_optimizer)
#         self.c, self.c_i = c.to(self.device), c_i.to(self.device)

#     def reset(self):
#         """Clean existing setups"""
#         self.datasize = None
#         self.trainloader = None
#         self.testloader = None
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
#         self.step_count = 0

#     def _scaffold_step(self, data, targets):
#         self.optimizer.zero_grad()

#         # forward pass
#         data, targets = data.to(self.device), targets.to(self.device)
#         logits = self.model(data)
#         loss = self.criterion(logits, targets)

#         # backward pass
#         loss.backward()
#         grad_batch = flatten_grads(self.model).detach().clone()
#         self.optimizer.zero_grad()

#         # add control variate
#         grad_batch = grad_batch - self.c_i + self.c
#         self.model = assign_grads(self.model, grad_batch)
#         self.optimizer.step()
#         self.step_count += 1

#     @torch.no_grad()
#     def _update_control_variate(self):

#         divisor = self.__get_divisor()

#         server_params = flatten_weights(self.dg_model)
#         local_params = flatten_weights(self.model)
#         param_move = server_params - local_params

#         c_i_plus = self.c_i.cpu() - self.c.cpu() + (divisor * param_move)
#         c_update_amount = c_i_plus - self.c_i.cpu()

#         return c_i_plus, c_update_amount