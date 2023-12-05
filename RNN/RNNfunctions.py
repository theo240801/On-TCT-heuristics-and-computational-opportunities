from io import open
import glob
import os
import torch.nn as nn

def findFiles(path): return glob.glob(path)

n_hidden = 128

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data_RNN/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn


import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor





def train_RNN(client_model, category_tensor_client, line_tensor_client):
    hidden = rnn.initHidden()

    client_model.zero_grad()

    for i in range(line_tensor_client.size()[0]):
        output, hidden = client_model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor_client)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return loss.item()

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def ClientTrainingData(num_clients):
    categories = []
    lines = []
    category_tensors = []
    line_tensors = []
    for i in range(num_clients):
        category_client = all_categories[i]
        lines_client = category_lines[category_client]
        category_tensor_client = torch.tensor([all_categories.index(category_client)], dtype=torch.long)
        categories.append(category_client)
        lines.append(lines_client)
        category_tensors.append(category_tensor_client)
        line_tensors_client = []
        for j in range(len(lines_client)):
            lines_tensor_client_j = lineToTensor(lines_client[j])
            line_tensors_client.append(lines_tensor_client_j)
        line_tensors.append(line_tensors_client)

    return categories, lines, category_tensors, line_tensors

learning_rate = 0.001 # If you set this too high, it might explode. If too low, it might not learn

criterion = nn.NLLLoss()

def train(model, category_tensor, line_tensor):
    hidden = model.initHidden()
    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def train_RNN(client_model, train_set, n_epochs=1):
    avg_loss = 0
    for epoch in range(n_epochs):
        loss = 0
        for j in range(len(train_set)):
            output, loss_i = train(client_model, train_set[j][1], train_set[j][0])
            loss += loss_i
        avg_loss += loss/len(train_set)
    return avg_loss/n_epochs

def evaluate(model, line_tensor):
    hidden = model.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output


import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# start = time.time()

# for iter in range(1, n_iters + 1):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     output, loss = train_RNN(rnn, category_tensor, line_tensor)
#     current_loss += loss

#     # Print ``iter`` number, loss, name and guess
#     if iter % print_every == 0:
#         guess, guess_i = categoryFromOutput(output)
#         correct = '✓' if guess == category else '✗ (%s)' % category
#         print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

#     # Add current loss avg to list of losses
#     if iter % plot_every == 0:
#         all_losses.append(current_loss / plot_every)
#         current_loss = 0