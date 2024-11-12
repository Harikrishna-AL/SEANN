import torch
from tqdm import tqdm
from torch import nn, optim


def get_excess_neurons(indices1, indices2, layer_sizes=[256, 128, 64]):
    # layer_sizes = [6,6,6]
    excess_neurons = []
    for i in range(len(indices1)):
        excess_neurons.append([j for j in indices2[i] if j not in indices1[i]])

    all_indices = [torch.arange(layer_sizes[i]) for i in range(len(layer_sizes))]

    if excess_neurons == [[], [], []]:
        for i in range(len(all_indices)):
            # delete the indicces present in indices2 from all_indices
            all_indices[i] = torch.tensor(
                [j for j in all_indices[i] if j not in indices2[i]]
            )
        return all_indices

    for i in range(len(indices1)):
        all_indices[i] = torch.tensor(
            [j for j in all_indices[i] if j not in excess_neurons[i]]
        )

    return all_indices


def get_merge_mask(mask1, mask2):
    merge_mask = []
    for i in range(len(mask1)):
        merge_mask.append(torch.logical_or(mask1[i], mask2[i]).int())
    return merge_mask


def calc_percentage_of_zero_grad(masks):
    total = 0
    zero = 0
    for mask in masks:
        total += mask.numel()
        zero += torch.sum(mask == 0).item()
    return (1 - zero / total) * 100


def forwardprop_and_backprop(
    model,
    lr,
    data_loader,
    continual=False,
    list_of_indexes=None,
    masks=None,
    scheduler=None,
    optimizer=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    loss_total = 0
    model.train()

    for i, (data, target) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        data = data.view(-1, 784)
        data, target = data.to(device), target.to(device)
        if not continual:
            output, list_of_indexes_out, masks = model(
                data, indexes=list_of_indexes, masks=masks
            )
            if i == len(data_loader) - 1:
                list_of_indexes = list_of_indexes_out

        else:

            output, list_of_indexes_out, masks_out = model(
                data, indexes=list_of_indexes, masks=masks
            )
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    scheduler.step()

    print("Avg loss: ", loss_total / len(data_loader))
    return list_of_indexes, masks, model, optimizer


# mask1 = [torch.tensor([1, 1, 0, 0, 1, 1]), torch.tensor([1, 1, 0, 0, 1, 1]), torch.tensor([0, 1, 0, 0, 0, 1])]
# mask2 = [torch.tensor([1, 1, 0, 0, 1, 1]), torch.tensor([1, 1, 0, 0, 1, 1]), torch.tensor([0, 1, 0, 1, 0, 1])]

# indices1 = [[2,3,4], [0,1,2,3,4,5], [2,4,5]]
# indices2 = [[2,3,4], [0,1,2,3,4,5], [2,4,5]]

# print(get_excess_neurons(indices1, indices2)) # [[], [], []]

# print(torch.arange(6))
