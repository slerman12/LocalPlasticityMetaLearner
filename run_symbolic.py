import math
import os
import time

import torch
import torchvision
from torch import Tensor
from torch.nn import init
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# HYPERPARAMS

n_epochs = 3
# batch_size_train = 64
batch_size_train = 1
batch_size_test = 1000
# learning_rate = 0.01
# momentum = 0.5
lr = 1e-3
optimizer_lr = 1
momentum = 0
# log_interval = 10
log_interval = 100

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# TRAIN/TEST DATA

train_data = torchvision.datasets.MNIST('./', train=True, download=True,
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
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=False,
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


class SelfOptimizingSequential(torch.nn.Module):
    def __init__(self, *layers, lr=0.01, E=F.binary_cross_entropy):
        super().__init__()
        self.lr = lr
        self.L = len(layers)
        self.alpha_L = None
        self.error = None

        # def E(y_pred, y_label, *vars, **kvargs):
        #     # MSE worse than cross entropy scaling; is there cross entropy with negatives?
        #     # return y_label - y_pred  # need to negate lr_coef
        #     return y_pred - y_label

        self.E = E
        for l_minus_1, layer in enumerate(layers):
            layer.l = l_minus_1 + 1
            layer.L = self.L
        self.layers = nn.ModuleList(layers)

    # TODO instead of checking for y_label, can have train and eval mode
    def P(self, x, y_label=None):
        alpha_l = x
        lr_coef = None
        n_l_minus_1_activated = None
        if self.alpha_L is not None and self.error is not None and y_label is not None:
            lr_coef = self.lr
            lr_coef = lr_coef * self.error
            lr_coef = lr_coef / ((1 - y_label) * lr_coef + 1)  # Why does this help? Even w/o w_l scaling? inc > dec?
            lr_coef = lr_coef * (1 - 2 * y_label)
        for layer in self.layers:
            alpha_l = layer(alpha_l, lr_coef, n_l_minus_1_activated)
            n_l_minus_1_activated = layer.n_l_activated
        self.alpha_L = alpha_l if self.alpha_L is None and y_label is not None else None
        rounding = 10e-7
        y_pred = torch.sigmoid(alpha_l - rounding)
        # y_pred = alpha_l
        self.error = self.E(y_pred, y_label, reduction='none') if self.error is None and y_label is not None else None
        return y_pred, alpha_l

    def forward(self, x, y_label=None):
        # TODO does this no_grad stop all autodiff overhead?
        with torch.no_grad():
            y_pred, alpha_L = self.P(x, y_label)
            return y_pred, alpha_L


# TODO pass in act function? (if not relu/clamp, needs to handle negative inputs)
class SelfOptimizingLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, is_output_layer=False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_output_layer = is_output_layer
        self.w_l = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.l = None
        self.L = None
        self.n_l_activated = None
        self.n_0_activated = torch.ones(in_features)[None, :]
        self.reset_parameters()

    def _forward(self, alpha_l_minus_1: Tensor, lr_coef=None, n_l_minus_1_activated=None) -> Tensor:
        # FORWARD PASS

        w_l = self.w_l
        z_l = F.linear(alpha_l_minus_1, w_l)
        clip = 10
        alpha_l = z_l if self.is_output_layer else torch.clamp(z_l, 0, clip)

        # SELF OPTIMIZATION

        if lr_coef is None:
            return alpha_l

        # Activated
        # TODO track if activations change
        self.n_l_activated = (alpha_l != 0).float()
        if n_l_minus_1_activated is None:
            if self.l == 1:
                n_l_minus_1_activated = self.n_0_activated
            else:
                n_l_minus_1_activated = (alpha_l_minus_1 != 0).float()

        # Compute delta (is matmul faster or slower than expand/repeat?)
        delta_l = torch.matmul(self.n_l_activated.unsqueeze(2), (n_l_minus_1_activated * lr_coef).unsqueeze(1))
        # w_l scaling (bottleneck)
        delta_l = delta_l * torch.abs(self.w_l)[None, :, :]  # doesn't this update in the wrong dir when alpha_L neg?
        delta_l = delta_l.mean(dim=0)
        # Hierarchical diminishing
        # delta_l = delta_l / self.l
        self.w_l.grad = delta_l

        # Resurrection (major bottleneck!)  # why does this also help w/o w scaling?
        r = 1e-5
        self.w_l[torch.abs(self.w_l) < r] = -torch.sign(self.w_l[torch.abs(self.w_l) < r]) * r

        return alpha_l

    def forward(self, alpha_l_minus_1: Tensor, lr_coef=None, n_l_minus_1_activated=None) -> Tensor:
        with torch.no_grad():
            return self._forward(alpha_l_minus_1, lr_coef, n_l_minus_1_activated)

    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.w_l, a=math.sqrt(5))
        init.normal_(self.w_l)
        # with torch.no_grad():
        #     if self.is_output_layer:
        #         self.w_l.copy_(torch.abs(self.w_l))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

# hypothesis: delta acc should go up with smaller lr TODO why doesn't it?
network = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(),
                        nn.Linear(64, 1), nn.Sigmoid())
network = SelfOptimizingSequential(SelfOptimizingLinear(784, 128),
                                   SelfOptimizingLinear(128, 64),
                                   SelfOptimizingLinear(64, 1, is_output_layer=True), lr=lr)
optimizer = optim.SGD(network.parameters(), lr=optimizer_lr, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []


def train(epoch):
    network.train()
    c1 = 0
    c2 = 0
    y_pred_delta = 0
    for batch_idx, (x, y_label) in enumerate(train_loader):
        x = torch.flatten(x, start_dim=1)
        y_label = y_label.float().unsqueeze(1)
        network(x, y_label)
        # network(x)

        optimizer.zero_grad()
        y_pred2, alpha_L_2 = network(x, y_label)
        # y_pred2 = network(x)
        loss = F.binary_cross_entropy(y_pred2, y_label)
        # With self-optimization, the backwards pass can be commented out:
        # loss.backward()
        optimizer.step()

        _, alpha_L_3 = network(x)

        if ((alpha_L_2[0] < alpha_L_3[0]) and y_label[0] == 1) or \
                ((alpha_L_2[0] > alpha_L_3[0]) and y_label[0] == 0) or \
                (alpha_L_2[0] == alpha_L_3[0] and loss < 1e-4):  # todo should be error[0], or should do batches
            c1 += 1
            # print(alpha_L_2, alpha_L_3, y_label)
        if alpha_L_2.item() < alpha_L_3.item():
            c2 += 1
            y_pred_delta += alpha_L_3[0].item() - alpha_L_2[0].item()
            # y_pred_delta += (alpha_L_3[0].item() - alpha_L_2[0].item()) / torch.abs(alpha_L_2[0]).item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            os.makedirs(os.path.dirname('./results/'), exist_ok=True)
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')
            print(" Correct y_pred Delta Sign: {}/{} ({:.3f}%)".format(c1, batch_idx + 1, 100. * c1 / (batch_idx + 1)))
            print(" Avg Pos y_pred Delta: {:.5f}".format(y_pred_delta/max(c2, 1)))

def test():
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y_label in test_loader:
            x = torch.flatten(x, start_dim=1)
            y_label = y_label.float().unsqueeze(1)
            y_pred, _ = network(x)
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
