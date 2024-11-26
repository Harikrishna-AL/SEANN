from utils import (
    get_excess_neurons,
    get_merge_mask,
    calc_percentage_of_zero_grad,
    forwardprop_and_backprop,
)
from model import NN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import get_data_separate_dynamic as get_data_separate
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


seed = 88  # verified
print("Seed: ", seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



label_groups = [
    [0, 1, 2, 3, 4],  
    [5, 6, 7, 8, 9]
]

train_loader_list, test_loader_list = get_data_separate(label_groups, batch_size=128)

list_of_indexes = [[], [], [],[]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
masks = [
    torch.ones(256).to(device),
    torch.ones(128).to(device),
    torch.ones(64).to(device),
    torch.ones(10).to(device),
]


original_model = NN(32*32, 10, indexes=list_of_indexes).to(device)
optimizer = optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

task_model_prev = original_model
task_indices_prev = list_of_indexes
indices = None
masks_list = []
task_model_list = []
continual = False
# create loop for all the tasks
for i in range(len(label_groups)):
    print(f"### Training Task {i+1} ###")
    for epoch in range(10):
        task_indices_i, task_masks_i, task_model_i, optimizer = forwardprop_and_backprop(
            task_model_prev,
            0.1,
            data_loader=train_loader_list[i],
            list_of_indexes=task_indices_prev,
            masks=masks,
            continual=continual,
            optimizer=None,
            scheduler=scheduler,
            indices_old=indices
        )
        task_model_prev = task_model_i
        task_indices_prev = task_indices_i

    if i==0:
        indices = []
        continual = True
    new_masks = []
    layer_sizes = [256, 128, 64, 10]
    for i in range(len(layer_sizes)):
        indices.append(
            torch.tensor(
                [j for j in range(layer_sizes[i]) if j not in task_indices_i[i]]
            ).to(device)
        )
        mask = torch.tensor(
            [1 if k in task_indices_i[i] else 0 for k in range(layer_sizes[i])]
        ).to(device)
        new_masks.append(mask)
    masks = new_masks

    masks_list.append(task_masks_i)
    task_model_list.append(task_model_prev)
    print("Task 1 indices: ", task_indices_i)
    print("Task 1 masks: ", task_masks_i)
    print("Percentage of frozen neurons: ", calc_percentage_of_zero_grad(task_masks_i))
            # print("percentage of zero gradients: ",calc_percentage_of_zero_grad(original_model))
            # print("loss: ", loss.item())
            # print("indices: ", indices)
            # print("masks: ", masks)
            # print("scalers: ", scalers)


# for i in range(10):
#     task1_indices, task1_masks, task1_model, optimizer = forwardprop_and_backprop(
#         original_model,
#         0.01,
#         data_loader_1,
#         list_of_indexes=list_of_indexes,
#         masks=masks,
#         optimizer=optimizer,
#         scheduler=scheduler,
#     )
#     list_of_indexes = task1_indices
#     # print("percentage of zero gradients: ",calc_percentage_of_zero_grad(original_model))

# indices = []
# new_masks = []
# layer_sizes = [256, 128, 64, 10]
# for i in range(len(layer_sizes)):
#     indices.append(
#         torch.tensor(
#             [j for j in range(layer_sizes[i]) if j not in task1_indices[i]]
#         ).to(device)
#     )
#     mask = torch.tensor(
#         [1 if k in task1_indices[i] else 0 for k in range(layer_sizes[i])]
#     ).to(device)
#     new_masks.append(mask)

# print("Task 1 indices: ", task1_indices)
# print("Task 1 masks: ", task1_masks)
# print("Percentage of frozen neurons: ", calc_percentage_of_zero_grad(task1_masks))

# # copy model 1 parameters to model 2
# # task2_model = NN(784, 10, indexes=indices).to(device)
# # task2_model.load_state_dict(task1_model.state_dict())

# print("### Task 2 ###")
# for i in range(10):
#     task2_indices, task2_masks, task2_model, optimizer = forwardprop_and_backprop(
#         task1_model,
#         0.1,
#         data_loader_2,
#         list_of_indexes=task1_indices,
#         masks=new_masks,
#         continual=True,
#         optimizer=None,
#         scheduler=scheduler,
#         indices_old=indices
#     )
# # print("Percentage of frozen neurons: ", calc_percentage_of_zero_grad(task2_masks))
# # print("percentage of zero gradients: ",calc_percentage_of_zero_grad(original_model))

# print("Task 2 indices: ", task2_indices)
# print("Task 2 masks: ", task2_masks)
# print("Percentage of frozen neurons: ", calc_percentage_of_zero_grad(task2_masks))

# test
# forwardprop_and_backprop(original_model, test_loader, list_of_indexes=list_of_indexes)
correct = 0
accuracies = []
original_model.eval()

# loop for all tests

for i in range(len(label_groups)):
    correct = 0
    task_model = task_model_list[i]
    print(f"### Testing Task {i+1}###")
    for data, target in test_loader_list[i]:
        data = data.view(-1, 32*32)
        data, target = data.to(device), target.to(device)
        output, scalers, indices, masks = task_model(data, masks=masks_list[i])
        # check the accuracy
        predicted = output.argmax(dim=1, keepdim=True)
        correct += predicted.eq(target.view_as(predicted)).sum().item()

    print(f"Accuracy for Task {i+1}: {100* correct/len(test_loader_list[i].dataset)}%")
    accuracies.append(100 * correct / len(test_loader_list[i].dataset))

# print("### Testing Task 1###")
# for data, target in test_loader_1:
#     data = data.view(-1, 32*32)
#     data, target = data.to(device), target.to(device)
#     output, scalers, indices, masks = task1_model(data, masks=task1_masks)
#     # check the accuracy
#     predicted = output.argmax(dim=1, keepdim=True)
#     correct += predicted.eq(target.view_as(predicted)).sum().item()

# print(f"Accuracy for Task 1: {100* correct/len(test_loader_1.dataset)}%")
# accuracies.append(100 * correct / len(test_loader_1.dataset))

# correct = 0
# print("### Testing Task 2###")
# for data, target in test_loader_2:
#     data = data.view(-1, 32*32)
#     data, target = data.to(device), target.to(device)
#     output, scalers, indices, masks = task2_model(data, masks=task2_masks)
#     # check the accuracy
#     predicted = output.argmax(dim=1, keepdim=True)
#     correct += predicted.eq(target.view_as(predicted)).sum().item()

# print(f"Accuracy for Task 2: {100* correct/len(test_loader_2.dataset)}%")
# accuracies.append(100 * correct / len(test_loader_2.dataset))


import matplotlib.pyplot as plt
import numpy as np

# hebbian_weights = task1_model.hebb_params[0].weight.data.cpu().numpy()
model_weights = task_model_list[-1].linear[0].weight.data.cpu().numpy()
model_weights1 = task_model_list[-1].linear[1].weight.data.cpu().numpy()

model_neurons = np.random.choice(256, 20)
model_neurons1 = np.random.choice(128, 20)
# select random 20 neurons
neurons = np.random.choice(256, 20)


# plt.figure(figsize=(20, 10))
# for i, neuron in enumerate(neurons):
#     plt.subplot(4, 5, i + 1)
#     plt.imshow(hebbian_weights[neuron].reshape(28, 28), cmap="gray")
#     plt.axis("off")

# plt.show()

plt.figure(figsize=(20, 10))
for i, neuron in enumerate(model_neurons):
    idx = neuron
    # idx = task2_indices[0][neuron]
    plt.subplot(4, 5, i + 1)
    plt.imshow(model_weights[idx].reshape(32, 32), cmap="gray")
    plt.axis("off")

plt.show()

plt.figure(figsize=(20, 10))
for i, neuron in enumerate(model_neurons1):
    idx = neuron
    # idx = task2_indices[1][neuron]
    plt.subplot(4, 5, i + 1)
    plt.imshow(model_weights1[idx].reshape(16, 16), cmap="gray")
    plt.axis("off")

plt.show()
