import os
import random
import time
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import Meta

# HYPERPARAMS

n_epochs = 3000
batch_size = 32
recurrence_steps = 1
batch_size_test = 1000
learning_rate = 1e-5
momentum = 0
log_interval = 1
log_interval *= recurrence_steps

torch.backends.cudnn.enabled = False

seed = 1
torch.manual_seed(seed)
random.seed(seed)

# TRAIN/TEST DATA

train_data = torchvision.datasets.MNIST('./', train=True, download=True,
                                        # torchvision transforms for some reason defy the random seed?
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
test_data = torchvision.datasets.MNIST('./', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
subset_indices = torch.nonzero((train_data.targets == 0) + (train_data.targets == 1)).view(-1)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                           sampler=SubsetRandomSampler(subset_indices))

subset_indices = torch.nonzero((test_data.targets == 0) + (test_data.targets == 1)).view(-1)
# print((test_data.test_labels == 0).sum(), (test_data.test_labels == 1).sum())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False,
                                          sampler=SubsetRandomSampler(subset_indices))

# PLOTTING

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
#
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])

# MODELS

# Standard MLP
# network = Meta.Sequential(nn.Linear(784, 128), nn.ReLU(),
#                         nn.Linear(128, 64), nn.ReLU(),
#                         nn.Linear(64, 1), nn.Sigmoid())
# recurrence_steps = 1
# log_interval /= recurrence_steps

# Meta MLP
# Seed Cell (with just SGD)
# seed_cell = Meta.SeedCell(num_layers=4, hidden_state_size=64, context_size=3)
# Seed Cell (evolutionary)
seed_cell = Meta.EvoSeedCell(num_layers=2, hidden_state_size=16, context_size=3, population_size=10, std=1)
# Single training stage only
# network = Meta.Sequential(Meta.Linear(seed_cell, 784, 128), nn.Tanh(),
#                           Meta.Linear(seed_cell, 128, 64), nn.Tanh(),
#                           Meta.Linear(seed_cell, 64, 1), nn.Sigmoid())
# Multiple training stages
network = Meta.MultiSequential(Meta.Linear(seed_cell, 784, 128), nn.Tanh(),
                               Meta.Linear(seed_cell, 128, 64), nn.Tanh(),
                               Meta.Linear(seed_cell, 64, 1), nn.Sigmoid(), num_sequentials=2)

# TRAINING

optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# optimizer = optim.Adam(network.parameters(), lr=learning_rate)

# allowing evolutionary seed cell to also be differentiated...
if hasattr(seed_cell, 'set_optim'):
    seed_cell.set_optim(optimizer)

train_losses = []
train_counter = []
test_losses = []
context = []
sequence = []


def train(epoch):
    network.train()

    # Stats collection
    loss_sum = _error_sum = correct = __correct = c1 = c2 = c3 = total = _total = y_pred_delta_pos = y_pred_delta_neg = 0

    _error = None

    for batch_idx, (x, y_label) in enumerate(train_loader):
        x = torch.flatten(x, start_dim=1)
        y_label = y_label.float().unsqueeze(1)

        batch_idx_total = batch_idx + len(train_loader) * (epoch - 1)

        if batch_idx_total > 0:
            sequence.append((x, y_label))

        # having multi sequentials operating at different stages of training
        if (batch_idx_total - 1) % recurrence_steps == 0 and random.random() > 0.95 and hasattr(network, 'reset_one'):
            network.reset_one()

        y_pred_original = network(x)
        y_pred = network(x, context, (batch_idx_total - 1) % recurrence_steps == 0)

        error = F.binary_cross_entropy(y_pred, y_label, reduction='none')
        loss = error.mean()
        loss_sum += loss

        # Meta-Optimization
        if batch_idx_total > 0 and batch_idx_total % recurrence_steps == 0:
            optimizer.zero_grad()
            _x = torch.cat([item[0] for item in sequence], dim=0)
            _y_label = torch.cat([item[1] for item in sequence], dim=0)
            _y_pred = network(_x)
            _error = F.binary_cross_entropy(_y_pred, _y_label, reduction='none')
            _loss = _error.mean() + loss_sum
            # comment/uncomment for SGD
            # _loss.backward()
            # optimizer.step()
            sequence.clear()
            loss_sum = 0

        # To test Standard MLP's deltas,
        # uncomment below
        # y_pred_original = y_pred
        # y_pred = network(x)

        # STATS

        y_pred_rounded = y_pred.data.round()
        _correct = y_pred_rounded.eq(y_label.data)
        correct += _correct.sum()
        total += y_pred.shape[0]

        if batch_idx_total > 0 and batch_idx_total % recurrence_steps == 0:
            _error_sum += _error.sum()
            _y_pred_rounded = _y_pred.data.round()
            ___correct = _y_pred_rounded.eq(_y_label.data)
            __correct += ___correct.sum()
            _total += _y_pred.shape[0]

        c1 += (((y_pred_original < y_pred) & (y_label == 1)) |
               ((y_pred_original > y_pred) & (y_label == 0)) |
               ((y_pred_original == y_pred) & _correct)).sum()
        _delta_sign_acc = c1 / total

        y_pred_delta_pos += \
            torch.nan_to_num((y_pred[y_pred_original < y_pred] - y_pred_original[y_pred_original < y_pred]).mean())
        c2 += (y_pred_original < y_pred).sum()
        y_pred_delta_neg -= \
            torch.nan_to_num((y_pred_original[y_pred_original > y_pred] - y_pred[y_pred_original > y_pred]).mean())
        c3 += (y_pred_original > y_pred).sum()

        y_label_prev, y_pred_prev, error_prev = y_label, y_pred, error
        # global_error = prev_error if _error is None else _error.view(recurrence_steps, -1, 1).mean(0).detach()
        # global_error = prev_error if _error is None else _error.view(-1, recurrence_steps, 1).mean(1).detach()
        # global_error = prev_error if _error is None else _error[0:-1:recurrence_steps].detach()
        # delta_sign_acc = torch.full_like(error_prev, _delta_sign_acc)
        context[:] = [y_label_prev, y_pred_prev, error_prev]
        assert len(context) == seed_cell.context_size

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), _error_sum / max(_total, 1)))
            print(" Accuracy: {}/{} ({:.0f}%)".format(__correct, _total, 100. * __correct / max(_total, 1)))
            train_losses.append(_error_sum / max(_total, 1))
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            os.makedirs(os.path.dirname('./Results/'), exist_ok=True)
            torch.save(network.state_dict(), './Results/model.pth')
            torch.save(optimizer.state_dict(), './Results/optimizer.pth')
            print(" Correct y_pred Delta Sign: {}/{} ({:.3f}%)".format(c1, total, 100. * c1 / total))
            print(" Avg Pos y_pred Delta: {:.10f}".format(y_pred_delta_pos / max(c2, 1)))
            print(" Avg Neg y_pred Delta: {:.10f}".format(y_pred_delta_neg / max(c3, 1)))

            # Stats collection
            correct = _error_sum = __correct = c1 = c2 = c3 = total = _total = y_pred_delta_pos = y_pred_delta_neg = 0


def test():
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y_label in test_loader:
            x = torch.flatten(x, start_dim=1)
            y_label = y_label.float().unsqueeze(1)
            y_pred = network(x)
            test_loss += F.binary_cross_entropy(y_pred, y_label).item()
            pred = y_pred.data.round()
            correct += pred.eq(y_label.data).sum()
            total += pred.shape[0]
    test_loss /= total
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))


start = time.time()
test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
print("\nTime: {}".format(time.time() - start))
