import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from data import get_data_separate
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


class GrowingNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(GrowingNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def grow_layer(self, layer_index):
        # increase the size of last layer
        if layer_index == -1:
            last_layer = self.layers[-1]
            new_layer = nn.Linear(last_layer.in_features, last_layer.out_features + 1)
            new_layer.weight.data[: last_layer.out_features, :] = last_layer.weight.data
            self.layers[-1] = new_layer

        else:
            layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]

            new_layer = nn.Linear(layer.in_features, layer.out_features + 1)
            new_next_layer = nn.Linear(
                next_layer.in_features + 1, next_layer.out_features
            )

            new_layer.weight.data[: layer.out_features, :] = layer.weight.data
            new_next_layer.weight.data[:, : next_layer.in_features] = (
                next_layer.weight.data
            )

            self.layers[layer_index] = new_layer
            self.layers[layer_index + 1] = new_next_layer


model = GrowingNetwork(784, 10, [128, 64, 32])
print(model)


class RL_GrowingAgent(nn.Module):
    def __init__(self, model, input_size, output_size, hidden_sizes):
        super(RL_GrowingAgent, self).__init__()
        self.model = model
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def reinforce(self, reward):
        pass

    def grow(self, layer_index):
        self.model.grow_layer(layer_index)


# rand_data = torch.randn(1, 784)

# output_size = 10
# for i in range(10):
#     layer_index = -1
#     model.grow_layer(layer_index=layer_index)

#     if layer_index == -1:
#         output_size += 1

#     target_data = torch.randn(1,output_size)
#     output = model(rand_data)

#     loss = nn.MSELoss()(output, target_data)
#     loss.backward()

#     print(loss.item())
#     print(output.shape, target_data.shape)
#     print(model)
