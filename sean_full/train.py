from utils import (
    get_excess_neurons,
    get_merge_mask,
    calc_percentage_of_zero_grad,
    forwardprop_and_backprop,
    merge_indices_and_masks,
)
from model import NN, RNNGate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import get_data_separate, get_domain_inc_data
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


seed = 10  # verified
print("Seed: ", seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

all_train_loaders, all_test_loaders = get_data_separate(
    batch_size=64, num_tasks=2, max_classes=10
)

list_of_indexes = [[], [], [], [], [], [], [], []]
layer_sizes = [32, 64, 128, 256, 256, 128, 64, 10]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_masks = [
    torch.ones(32).to(device),
    torch.ones(64).to(device), 
    torch.ones(128).to(device),
    torch.ones(256).to(device), 
    torch.ones(256).to(device),
    torch.ones(128).to(device),
    torch.ones(64).to(device),
    torch.ones(10).to(device),
]

task_model = NN(256, 10, indexes=list_of_indexes).to(device)
rnn_gate = RNNGate(784, 10, 2).to(device)

all_model_params = task_model.parameters()
# all_model_params.extend(rnn_gate.parameters())
optimizer = optim.SGD(all_model_params, lr=0.1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

all_task_indices = list_of_indexes
all_task_masks = []

all_masks = []

for t in range(len(all_train_loaders)):

    print("### Task ", t + 1, " ###")
    for i in range(10):
        task_indices, task_masks, task_model, optimizer = forwardprop_and_backprop(
            task_model,
            0.1,
            all_train_loaders[t],
            list_of_indexes=list_of_indexes,
            masks=task_masks,
            optimizer=optimizer if t == 0 else None,
            scheduler=scheduler,
            task_id=t + 1,
            rnn_gate=rnn_gate,
            continual=False if t == 0 else True,
            indices_old=None if t == 0 else indices_old,
            prev_parameters=None if t == 0 else prev_parameters,
        )

    all_task_indices, all_task_masks = merge_indices_and_masks(
        all_task_indices, task_indices, all_task_masks, task_masks
    )  # merge the indices of the current task with the previous tasks

    indices_old = []
    masks = []

    print(task_indices)
    print(task_masks)
    for i in range(len(layer_sizes)):
        indices_old.append(
            torch.tensor(
                [j for j in range(layer_sizes[i]) if j not in all_task_indices[i]]
            ).to(device)
        )
        mask = torch.tensor(
            [1 if k in all_task_indices[i] else 0 for k in range(layer_sizes[i])]
        ).to(device)
        masks.append(mask)

    all_masks.append(task_masks)

    print("Task ", t + 1, " indices: ", task_indices)
    print("Task ", t + 1, " masks: ", task_masks)
    print("Percentage of frozen neurons", calc_percentage_of_zero_grad(all_task_masks))
    task_masks = masks
    prev_parameters = task_model.linear

accuracies = []
task_model.eval()

ones_mask = [
    torch.ones(32).to(device),
    torch.ones(64).to(device), 
    torch.ones(128).to(device),
    torch.ones(256).to(device), 
    torch.ones(256).to(device),
    torch.ones(128).to(device),
    torch.ones(64).to(device),
    torch.ones(10).to(device),
]

def find_task_id(generated_mask, learned_masks):
    #check similarity between the two masks
    scores = []
    for i in range(len(learned_masks)):
        score = 0
        for j in range(len(generated_mask[0])):
            score += torch.max(torch.sum(
                torch.logical_and(generated_mask[0][j], learned_masks[i][j])
            ) / torch.sum(generated_mask[0][j]), 
                               torch.sum(torch.logical_and(generated_mask[1][j], learned_masks[i][j])) / torch.sum(generated_mask[1][j]))
        scores.append(score.item())
    # print("scores: ", scores)    
    # return the index of the mask with the highest similarity
    return scores.index(max(scores))


def get_task_id_from_weights(weights, masks, data):
    scores = []
    data_mean = torch.mean(data, dim=0)
    for i in range(len(masks)):
        weight = F.normalize(weights[0].weight,p=2, dim=1)
        weight_mean = torch.mean(
            weight *  masks[i][0].T, dim=0)
 
        score = torch.sum(torch.abs(data_mean - weight_mean))
        scores.append(score.item())
    
    return scores.index(min(scores))
        
        
        
for t in range(len(all_test_loaders)):
    correct = 0
    test_loader = all_test_loaders[t]
    print("### Testing Task ", t + 1, " ###")
    for i, (data, target) in enumerate(test_loader):
        data = data.view(-1, 784).to(device)
        target = target.to(device)
        # inference_masks = []
        # for m in range(len(all_masks)):
        #     output, scalers, indices, masks, _ = task_model(
        #         data, masks=all_masks[m], indices_old=[None] * len(masks)
        #     )
        #     inference_masks.append(masks)
            
        # print("masks: ", masks)
        # mask_id = find_task_id(inference_masks, all_masks)
        
        # if i == 0:
        #     activations = []
        #     layer1 = task_model.linear[0].weight
        #     # normalize the weights
        #     layer1 = F.normalize(layer1, p=2, dim=1)
        #     print("layer1: ", layer1.shape)
        #     print("data: ", data.shape)
        #     activation1 = data @ layer1.T
        #     for j in range(len(all_masks)):
        #         activation1 = F.relu(activation1 * all_masks[j][0])
        #         activations.append(
        #             torch.sum(activation1, dim=1) / torch.sum(all_masks[j][0])
        #         )
        #     # print("activations: ", activations)
        #     activations = torch.stack(activations, dim=0)  # (2, batch_size)
        #     mask_id = torch.argmax(activations, dim=0)  # shape: (batch_size,)
        #     print("mask_id: ", mask_id)
        # mask_id = get_task_id_from_weights(task_model.linear, all_masks, data)
        output, scalers, indices, masks, _ = task_model(
            data, masks=all_masks[t], indices_old=[None] * len(masks)
        )
        
        # check the accuracy
        predicted = output.argmax(dim=1, keepdim=True)
        correct += predicted.eq(target.view_as(predicted)).sum().item()
    print(f"Accuracy for Task {t + 1}: {100 * correct / len(test_loader.dataset):.2f}%")
    accuracies.append(100 * correct / len(test_loader.dataset))

# print("### Testing both Tasks using entropy as gating mechanism ###")
# for data, target in test_loader:
#     data = data.view(-1, 784).to(device)
#     target = target.to(device)

#     outputs = []
#     entropies = []

#     for mask in all_masks:
#         # Forward pass through the subnetwork defined by the mask
#         output, scalers, indices, masks, _ = original_model(
#             data, masks=mask, indices_old=[None] * len(mask)
#         )
#         outputs.append(output)

#         prob = output
#         # prob = F.softmax(output, dim=1)
#         entropy = -torch.sum(
#             prob * torch.log(prob + 1e-10), dim=1
#         )  # shape: (batch_size,)
#         entropies.append(entropy)

#     entropies = torch.stack(entropies, dim=0)  # (2, batch_size)
#     outputs = torch.stack(outputs, dim=0)  # (2, batch_size, num_classes)

#     # Choose output from network with minimum entropy for each sample
#     min_entropy_indices = torch.argmin(entropies, dim=0)  # shape: (batch_size,)

#     batch_size = data.size(0)
#     final_preds = torch.zeros(batch_size, dtype=torch.long, device=device)

#     for i in range(batch_size):
#         selected_output = outputs[min_entropy_indices[i], i]
#         final_preds[i] = selected_output.argmax()

#     correct += final_preds.eq(target).sum().item()

# print(f"Accuracy for both Tasks: {100 * correct / len(test_loader.dataset):.2f}%")


# print("#### Testing both Tasks using RNN as gating mechanism ####")
# correct = 0
# for data, target in test_loader:
#     data = data.view(-1, 784).to(device)
#     target = target.to(device)

#     gate_out = rnn_gate(data)
#     gate_outs = torch.argmax(gate_out, dim=1)  # shape: (batch_size,)
#     outputs = []
#     for i in range(len(gate_outs)):
#         data_in = data[i].unsqueeze(0)
#         data_in = data_in.view(-1, 784).to(device)
#         output, scalers, indices, masks, _ = original_model(data_in, masks=all_masks[gate_outs[i]], indices_old=[None]*len(task1_masks))
#         # output = output.squeeze(1)
#         predicted = output.argmax(dim=1, keepdim=True)
#         predicted = predicted.squeeze(1)
#         predicted = predicted.squeeze(0)
#         outputs.append(predicted)
#     outputs = torch.stack(outputs, dim=0)      # (2, batch_size, num_classes)
#     # final_preds = outputs.argmax(dim=1, keepdim=True)
#     # print(outputs)
#     correct += final_preds.eq(target.view_as(outputs)).sum().item()
# print(f"Accuracy for both Tasks: {100 * correct / len(test_loader.dataset):.2f}%")

# correct = 0
# for i, (data, target) in enumerate(test_loader):
#     data = data.view(-1, 784).to(device)
#     target = target.to(device)

#     activations = []

#     if i == 0:
#         layer1 = task1_model.linear[0]
#         activation1 = layer1(data)
#         for j in range(len(all_masks)):
#             activation1 = F.relu(activation1 * all_masks[j][0])
#             activations.append(
#                 torch.sum(activation1, dim=1) / torch.sum(all_masks[j][0])
#             )
#         activations = torch.stack(activations, dim=0)
#         print("argmax of avg activation: ", torch.argmax(activations, dim=0))
#         print("activations: ", activations)
#         print("Target: ", target)

# correct = 0
# print("### Testing Task 1###")
# task_id = 1
# for data, target in test_loader_1:
#     data = data.view(-1, 784)
#     data, target = data.to(device), target.to(device)
#     output, scalers, indices, masks, _ = task1_model(
#         data, masks=task1_masks, indices_old=[None] * len(indices)
#     )
#     # check the accuracy
#     predicted = output.argmax(dim=1, keepdim=True)
#     correct += predicted.eq(target.view_as(predicted)).sum().item()

# print(f"Accuracy for Task 1: {100* correct/len(test_loader_1.dataset)}%")
# accuracies.append(100 * correct / len(test_loader_1.dataset))

# # task2_masks = get_merge_mask(task1_masks, task2_masks)

# correct = 0
# print("### Testing Task 2###")
# task_id = 2
# for data, target in test_loader_2:
#     data = data.view(-1, 784)
#     data, target = data.to(device), target.to(device)
#     output, scalers, indices, masks, _ = task2_model(
#         data, masks=task2_masks, indices_old=[None] * len(indices)
#     )
#     # check the accuracy
#     predicted = output.argmax(dim=1, keepdim=True)
#     # target = target % 5
#     correct += predicted.eq(target.view_as(predicted)).sum().item()

# print(f"Accuracy for Task 2: {100* correct/len(test_loader_2.dataset)}%")
# accuracies.append(100 * correct / len(test_loader_2.dataset))

# layer1 = task2_model.linear[0]
# #find out the principal direction of the weights of both the subnetworks
# # layer1_task1 = torch.where(task1_masks[0] == 1, layer1.weight.T, torch.zeros_like(layer1.weight.T))
# # layer1_task2 = torch.where(task2_masks[0] == 1, layer1.weight.T, torch.zeros_like(layer1.weight.T))
# # layer1_task2 = layer1.weight.T * task2_masks[0]
# layer1_task1 = layer1.weight[task1_masks[0].squeeze(0) == 1]
# layer1_task2 = layer1.weight[task2_masks[0].squeeze(0) == 1]
# print(layer1_task1)
# print(layer1_task2)
# print(layer1_task1.shape)
# print(layer1_task2.shape)
# #plot the principal direction of the weights of both the subnetworks
# import numpy as np
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(layer1_task1.cpu().detach().numpy())
# layer1_task1 = pca.transform(layer1_task1.cpu().detach().numpy())
# pca.fit(layer1_task2.cpu().detach().numpy())
# layer1_task2 = pca.transform(layer1_task2.cpu().detach().numpy())

# import matplotlib.pyplot as plt
# plt.scatter(layer1_task1[:, 0], np.arange(len(layer1_task1)),color='red', label='Task 1')
# plt.scatter(layer1_task2[:, 0], np.arange(len(layer1_task2)),color='blue', label='Task 2')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('Principal Component Analysis of Task 1 and Task 2 Weights')
# plt.legend()
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# hebbian_weights = task1_model.hebb_params[0].weight.data.cpu().numpy()
# model_weights = task2_model.linear[0].weight.data.cpu().numpy()
# model_weights1 = task2_model.linear[1].weight.data.cpu().numpy()

# model_neurons = np.random.choice(256, 20)
# model_neurons1 = np.random.choice(128, 20)
# # select random 20 neurons
# neurons = np.random.choice(256, 20)


# plt.figure(figsize=(20, 10))
# for i, neuron in enumerate(neurons):
#     plt.subplot(4, 5, i + 1)
#     plt.imshow(hebbian_weights[neuron].reshape(28, 28), cmap="gray")
#     plt.axis("off")

# plt.show()

# plt.figure(figsize=(20, 10))
# for i, neuron in enumerate(model_neurons):
#     idx = neuron
#     # idx = task2_indices[0][neuron]
#     plt.subplot(4, 5, i + 1)
#     plt.imshow(model_weights[idx].reshape(28, 28), cmap="gray")
#     plt.axis("off")

# plt.show()

# plt.figure(figsize=(20, 10))
# for i, neuron in enumerate(model_neurons1):
#     idx = neuron
#     # idx = task2_indices[1][neuron]
#     plt.subplot(4, 5, i + 1)
#     plt.imshow(model_weights1[idx].reshape(16, 16), cmap="gray")
#     plt.axis("off")

# plt.show()
