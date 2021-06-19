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

n_epochs = 3
batch_size = 64
# batch_size = 64
batch_size_test = 1000
learning_rate = 1e-4
momentum = 0
# momentum = 0.5
log_interval = 100 // batch_size

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

# Meta MLP
network = Meta.Sequential(Meta.Linear(784, 128), nn.Tanh(),
                          Meta.Linear(128, 64), nn.Tanh(),
                          Meta.Linear(64, 1), nn.Sigmoid())

# TRAINING

optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# optimizer = optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
train_counter = []
test_losses = []


def train(epoch):
    network.train()

    # Stats collection
    c1 = 0
    c2 = 0
    c3 = 0
    y_pred_delta_pos = 0
    y_pred_delta_neg = 0

    for batch_idx, (x, y_label) in enumerate(train_loader):
        x = torch.flatten(x, start_dim=1)
        y_label = y_label.float().unsqueeze(1)

        if batch_idx > 0:
            # Don't want y_pred_original (which is just meant for stats after batch_idx 0) to update network.prev_input
            network.eval()
        y_pred_original = network(x)
        network.train()

        if batch_idx > 0:
            optimizer.zero_grad()
            y_pred = network(x, prev_y_label, prev_y_pred, prev_error)
        else:
            y_pred = y_pred_original

        error = F.binary_cross_entropy(y_pred, y_label, reduction='none')
        loss = error.mean()

        if batch_idx > 0:
            loss.backward()
            optimizer.step()

        prev_y_label, prev_y_pred, prev_error = y_label, y_pred, error

        # To test Standard MLP's deltas,
        # uncomment below
        # y_pred_original = y_pred
        # y_pred = network(x)

        # STATS

        # if ((y_pred_original[0] < y_pred[0]) and y_label[0] == 1) or \
        #         ((y_pred_original[0] > y_pred[0]) and y_label[0] == 0) or \
        #         (y_pred_original[0] == y_pred[0] and error[0] < 1e-4):
        #     c1 += 1

        c1 += (((y_pred_original < y_pred) & (y_label == 1)) |
               ((y_pred_original > y_pred) & (y_label == 0)) |
               ((y_pred_original == y_pred) & (error < 1e-4))).sum()

        # if y_pred_original[0].item() < y_pred[0].item():
        #     c2 += 1
        #     y_pred_delta_pos += y_pred[0].item() - y_pred_original[0].item()
        # if y_pred_original[0].item() > y_pred[0].item():
        #     c3 += 1
        #     y_pred_delta_neg -= y_pred_original[0].item() - y_pred[0].item()

        y_pred_delta_pos += \
            torch.nan_to_num((y_pred[y_pred_original < y_pred] - y_pred_original[y_pred_original < y_pred]).mean())
        c2 += (y_pred_original < y_pred).sum()
        y_pred_delta_neg -= \
            torch.nan_to_num((y_pred_original[y_pred_original > y_pred] - y_pred[y_pred_original > y_pred]).mean())
        c3 += (y_pred_original > y_pred).sum()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch-1) * len(train_loader.dataset)))
            os.makedirs(os.path.dirname('./Results/'), exist_ok=True)
            torch.save(network.state_dict(), './Results/model.pth')
            torch.save(optimizer.state_dict(), './Results/optimizer.pth')
            total = batch_size * (batch_idx + 1)
            print(" Correct y_pred Delta Sign: {}/{} ({:.3f}%)".format(c1, total, 100. * c1 / total))
            print(" Avg Pos y_pred Delta: {:.5f}".format(y_pred_delta_pos / max(c2, 1)))
            print(" Avg Neg y_pred Delta: {:.5f}".format(y_pred_delta_neg / max(c3, 1)))


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
