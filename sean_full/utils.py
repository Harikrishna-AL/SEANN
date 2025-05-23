import torch
from tqdm import tqdm
from torch import nn, optim


def merge_indices_and_masks(
    all_task_indices, task_indices, all_task_masks, task_masks, max_classes=10
):
    # take union of all_task_indices and task_indices
    merge_mask = []
    merge_indices = []
    layer_sizes = [256, 128, 64, max_classes]
    
    for i in range(len(all_task_masks)):
        if all_task_masks != []:
            merge_mask.append(torch.logical_or(all_task_masks[i], task_masks[i]))

        if all_task_indices != [[], [], [], []]:
            common = torch.isin(
                all_task_indices[i], task_indices[i]
            )
            common = all_task_indices[i][common]
            merge_indices.append(
                common
            )
    if all_task_masks == []:
        merge_mask = task_masks
    if all_task_indices == [[], [], [], []]:
        merge_indices = task_indices
        
    return merge_indices, merge_mask

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
    for i in range(len(masks)-1):
        total += masks[i].numel()
        zero += torch.sum(masks[i] == 0).item()
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
    task_id=None,
    prev_parameters=None,
    rnn_gate=None,
    max_classes=10,
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
        data = data.view(-1, 784)
        data, target = data.to(device), target.to(device)
        
        one_hot_target = torch.zeros(target.size(0), max_classes).to(device)
        one_hot_target.scatter_(1, target.view(-1, 1), 1)
        if not continual:
            indices_old = [None] * len(list_of_indexes)
  
            output, scalers, list_of_indexes, masks, common_indices = model(
                data, scalers, indexes=list_of_indexes, masks=masks, indices_old = indices_old, target=one_hot_target, selection_method="hebbian",
            )
  
        else:
            output, scalers, list_of_indexes, masks, common_indices = model(
                data,
                scalers,
                indexes=list_of_indexes,
                masks=masks,
                indices_old=indices_old,
                target=one_hot_target,
                selection_method="hebbian",
            )
        
        # if task_id is not None:
        #     output = output[:, 5*(task_id-1):5*task_id]
        #     target = target % 5
            
        loss = criterion(output, target) 
        # print(common_indices)
        if common_indices is not None:
            #add ewc loss to the main loss
            ewc_loss = 0
            for i in range(len(common_indices)):
                ewc_loss += torch.sum(
                    torch.pow(
                        model.linear[i].weight[common_indices[i]]
                        - prev_parameters[i][common_indices[i]],
                        2,
                    )
                )
            loss += 0.5 * ewc_loss
        
                    
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    scheduler.step()

    print("Avg loss: ", loss_total / len(data_loader))
    return list_of_indexes, masks, model, optimizer
