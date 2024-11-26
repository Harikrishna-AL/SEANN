import torch
from tqdm import tqdm
from torch import nn, optim


def get_excess_neurons(indices1, indices2, layer_sizes=[256, 128, 64]):
    """
    Identifies neurons in each layer that are present in indices2 but not in indices1.

    Args:
        indices1 (list of lists): Indices of neurons selected for a particular task or layer.
        indices2 (list of lists): Indices of neurons for comparison with indices1.
        layer_sizes (list, optional): List of neuron counts per layer.

    Returns:
        list of torch.Tensor: List of indices representing neurons not present in either indices1 or indices2.
    """

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
    """
    Merges two sets of binary masks using logical OR operation.

    Args:
        mask1 (list of torch.Tensor): First list of masks.
        mask2 (list of torch.Tensor): Second list of masks.

    Returns:
        list of torch.Tensor: List of merged masks, where each mask is the result of logical OR operation.
    """

    merge_mask = []
    for i in range(len(mask1)):
        merge_mask.append(torch.logical_or(mask1[i], mask2[i]).int())
    return merge_mask


def calc_percentage_of_zero_grad(masks):
    """
    Calculates the percentage of neurons with non-zero gradients in the given masks.

    Args:
        masks (list of torch.Tensor): List of masks for each layer.

    Returns:
        float: Percentage of neurons with non-zero gradients.
    """

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
    indices_old = None,
):
    """
    Performs forward and backward propagation over a dataset with optional continual learning.

    Args:
        model (nn.Module): Neural network model.
        lr (float): Learning rate for optimizer.
        data_loader (DataLoader): DataLoader for the training data.
        continual (bool, optional): Flag indicating whether continual learning is applied.
        list_of_indexes (list, optional): List of indexes for selective neuron training.
        masks (list, optional): List of masks for each layer.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        optimizer (torch.optim.Optimizer, optional): Optimizer for the model.

    Returns:
        tuple: Updated list of indexes, masks, model, and optimizer after training.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    loss_total = 0
    model.train()
    scalers = None
    for i, (data, target) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        data = data.view(-1, 32*32)
        # print(data.shape, target.shape)
        data, target = data.to(device), target.to(device)
        
        one_hot_target = torch.zeros(target.size(0), 10).to(device)
        one_hot_target.scatter_(1, target.view(-1, 1), 1)
        if not continual:
            if scalers is not None:
                output, scalers, list_of_indexes, masks = model(
                    data, scalers, indexes=list_of_indexes, masks=masks, target=one_hot_target
                )
            else:
                output, scalers, list_of_indexes, masks = model(
                    data, indexes=list_of_indexes, masks=masks, target=one_hot_target
                )

        else:
            # if i == 0:
            #     list_of_indexes_out = list_of_indexes

            output, scalers, list_of_indexes, masks = model(
                data,
                scalers,
                indexes=list_of_indexes,
                masks=masks,
                indices_old=indices_old,
                target=one_hot_target,
            )

            # if i == len(data_loader) - 1:
            #     list_of_indexes = list_of_indexes_out

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    print("scalers", scalers)

    scheduler.step()

    print("Avg loss: ", loss_total / len(data_loader))
    return list_of_indexes, masks, model, optimizer
