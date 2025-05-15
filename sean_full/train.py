from utils import (
    get_excess_neurons,
    get_merge_mask,
    calc_percentage_of_zero_grad,
    forwardprop_and_backprop,
    merge_indices_and_masks,
    compute_average_accuracy,
    compute_forward_transfer,
)
from model import NN, RNNGate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import (
    get_data_separate,
    get_domain_inc_data,
    get_cifar10_data,
    get_cifar100_data,
)
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import argparse


# seed = 10  # verified

def train(seed, num_tasks=2, batch_size=128, data_type="mnist", output_size=10, lr=0.1, epochs=10):
    seeds = [seed]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    
    
    if data_type == "mnist":
        all_train_loaders, all_test_loaders = get_data_separate(
            batch_size=batch_size, num_tasks=num_tasks, max_classes=output_size
        )
    elif data_type == "cifar10":
        all_train_loaders, all_test_loaders = get_cifar10_data(
            batch_size=batch_size, num_tasks=num_tasks, max_classes=output_size
        )

    elif data_type == "cifar100":
        all_train_loaders, all_test_loaders = get_cifar100_data(
            batch_size=batch_size, num_tasks=num_tasks, max_classes=output_size
        )
    else:
        raise ValueError("Invalid data type. Choose from 'mnist', 'cifar10', or 'cifar100'.")
    
    all_accuracies = []


    for seed in seeds:
        print("Seed: ", seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # layer_sizes = [32, 64, 128, 256, 256, 1024, 512, output_size]
        if data_type == "mnist":
            layer_sizes = [256, 128, 64, output_size]
        elif data_type == "cifar10":
            # layer_sizes = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, output_size]
            layer_sizes = [32, 64, 128,128, 256, 256, 1024, 512, output_size]
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        task_masks = [
            torch.ones(i).to(device) for i in layer_sizes
        ]  # create a mask for each layer

        list_of_indexes = [[]] * len(layer_sizes)
        
        input_size = 256*4*4 if data_type == "cifar10" else 28 * 28

        task_model = NN(input_size, output_size, indexes=list_of_indexes, data=data_type, num_tasks = num_tasks).to(device)
        # rnn_gate = RNNGate(784, 100, 2).to(device)

        all_model_params = task_model.parameters()
        # all_model_params.extend(rnn_gate.parameters())
        optimizer = optim.SGD(all_model_params, lr=lr, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        all_task_indices = list_of_indexes
        all_task_masks = []

        all_masks = []
        mat_size = len(all_train_loaders)
        accuracy_matrix = torch.zeros(mat_size, mat_size).to(device)


        for t in range(len(all_train_loaders)):
            task_model.train()
            print("### Task ", t + 1, " ###")
            for i in range(epochs):
                task_indices, task_masks, task_model, optimizer = forwardprop_and_backprop(
                    task_model,
                    0.1,
                    all_train_loaders[t],
                    list_of_indexes=list_of_indexes,
                    masks=task_masks,
                    optimizer=optimizer if t == 0 else None,
                    scheduler=scheduler,
                    task_id=t + 1,
                    # rnn_gate=rnn_gate,
                    continual=False if t == 0 else True,
                    indices_old=None if t == 0 else indices_old,
                    prev_parameters=None if t == 0 else prev_parameters_list,
                    output_size=output_size,
                    epoch = i,
                    data_type=data_type,
                )

            all_task_indices, all_task_masks = merge_indices_and_masks(
                all_task_indices,
                task_indices,
                all_task_masks,
                task_masks,
                layer_sizes=layer_sizes,
            )  # merge the indices of the current task with the previous tasks

            indices_old = []
            masks = []

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
            frozen_neurons = calc_percentage_of_zero_grad(all_task_masks)
            print(
                "Percentage of frozen neurons", frozen_neurons
            )
            task_masks = masks
            prev_parameters = task_model.layers
            prev_parameters_list = {}
            
            if frozen_neurons > 85.0:
                layer_sizes_new = task_model.grow_layers()
                task_model.to(device)
                task_masks = [
                    torch.ones(layer_sizes_new[i]).to(device)
                    for i, mask in enumerate(task_masks)
                ]
                
                layer_sizes_diff = [layer_sizes_new[i] - layer_sizes[i] for i in range(len(layer_sizes))]
                print(all_task_masks[0].shape)
                all_task_masks = [
                    torch.cat((all_task_masks[i], torch.zeros(layer_sizes_diff[i]).unsqueeze(0).to(device)), dim=1) for i in range(len(all_task_masks))
                ]
                # update all task mask size
                
                layer_sizes = layer_sizes_new
                
                all_masks = [
                    [
                        torch.cat((all_masks[i][j], torch.zeros(layer_sizes_diff[j]).unsqueeze(0).to(device)), dim=1)
                        for j in range(len(all_masks[i]))
                    ] for i in range(len(all_masks))
                ]

                #udpate the masks size
                
            for i in range(len(prev_parameters)):
                layer = prev_parameters[i]
                if (
                    isinstance(layer, (nn.Linear, nn.Conv2d))
                ) and i < len(prev_parameters) - 1:
                    prev_parameters_list[i] = layer.weight.data.clone()
            
            print("Evaluating Task ", t + 1)
            print("length of all task masks", len(all_task_masks))
            print("length of all masks", len(all_masks))
            print("length of all test loaders", len(all_test_loaders))
            
            task_model.eval()

            for j in range(t + 1):
                correct = 0
                total = 0
                for data, target in all_test_loaders[j]:
                    if data_type == "mnist":
                        data = data.view(-1, 28 * 28).to(device)
                    elif data_type == "cifar10":
                        data = data.view(-1, 3, 32, 32).to(device)
                    target = target.to(device)
                    output, _, _, masks, _ = task_model(data, masks=all_masks[j], indices_old=[None]*len(masks))
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                acc = 100.0 * correct / total
                accuracy_matrix[t, j] = acc
                print(f"Accuracy on Task {j+1} after training Task {t+1}: {acc:.2f}%")
                # print("Accuracy Matrix after task", t + 1)
                # print(accuracy_matrix[:t + 1, :t + 1].cpu().numpy())
            
            if t < num_tasks - 1:
                print(f"Estimating FWT: Task {t+1} → Task {t+2} (before training Task {t+2})")
                task_indices_next, task_masks_next, _, _ = forwardprop_and_backprop(
                    model=task_model, 
                    lr=0.1, 
                    data_loader=all_train_loaders[t + 1], 
                    list_of_indexes=list_of_indexes, 
                    masks=[torch.ones_like(m) for m in task_masks],
                    optimizer=None, 
                    scheduler=None, 
                    task_id=t+2, 
                    continual=True,
                    indices_old=indices_old, 
                    prev_parameters=prev_parameters_list,
                    output_size=output_size, 
                    epoch=0, 
                    data_type=data_type,
                    compute_only_mask=True,
                )

                correct = 0
                total = 0
                for data, target in all_test_loaders[t + 1]:
                    data = data.view(data.size(0), -1).to(device) if data_type == "mnist" else data.to(device)
                    target = target.to(device)
                    output, _, _, _, _ = task_model(data, masks=task_masks_next, indices_old=[None]*len(task_masks_next))
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                acc = 100.0 * correct / total
                accuracy_matrix[t, t + 1] = acc
                print(f"FWT A[{t}][{t+1}] = {acc:.2f}%")

                print("Accuracy Matrix after task", t + 1)
                print(accuracy_matrix[:t + 1, :t + 1].cpu().numpy())

        # accuracies = []
        # task_model.eval()

        
        # accuracy_matrix = torch.zeros(num_tasks, num_tasks).to(device)
        # print("testing all tasks")

        # for j in range(t + 1):
        #     correct = 0
        #     test_loader = all_test_loaders[j]
        #     print("Testing Task ", j + 1)

        #     for data, target in test_loader:
        #         if data_type == "mnist":
        #             data = data.view(-1, 28 * 28).to(device)
        #         elif data_type == "cifar10":
        #             data = data.view(-1, 3, 32, 32).to(device)
        #         target = target.to(device)

        #         output, _, _, masks, _ = task_model(
        #             data, masks=all_masks[j], indices_old=[None] * len(masks)
        #         )
        #         predicted = output.argmax(dim=1, keepdim=True)
        #         correct += predicted.eq(target.view_as(predicted)).sum().item()

        #     acc = 100.0 * correct / len(test_loader.dataset)
        #     accuracy_matrix[t, j] = acc
        #     print(f"Accuracy on Task {j+1} after training Task {t+1}: {acc:.2f}%")
        #     print("popopopo")

        # all_accuracies.append(torch.tensor(accuracies).to(device))

    # accuracy_matrix[i, j]: accuracy on task j after training task i
    # accuracy_matrix = torch.zeros((len(all_accuracies[0]), len(all_accuracies[0])))

    # Fill accuracy matrix: after training task i, accuracy on task j
    # for i in range(len(all_accuracies[0])):
    #     for j in range(i + 1):
    #         accuracy_matrix[i, j] = all_accuracies[0][j]

    print("len of all task masks", len(all_task_masks))

    print("Accuracy Matrix after task", t + 1)
    print(accuracy_matrix[:t + 1, :t + 1].cpu().numpy())    

    avg_acc = compute_average_accuracy(accuracy_matrix.numpy())
    fwt = compute_forward_transfer(accuracy_matrix.numpy(), random_baseline=10.0)

    print(f"Average Accuracy: {avg_acc:.2f}%")
    print(f"Forward Transfer (FWT): {fwt:.2f}%")

    # Optional: plot the accuracy matrix heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        accuracy_matrix[:num_tasks, :num_tasks].cpu().numpy(),
        annot=True, fmt=".2f", cmap="Blues",
        xticklabels=[f"T{i+1}" for i in range(num_tasks)],
        yticklabels=[f"T{i+1}" for i in range(num_tasks)]
    )
    plt.xlabel("Evaluation Task")
    plt.ylabel("After Training Task")
    plt.title("Continual Learning Accuracy Matrix")
    plt.show()
    

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--seed", type=int, default=10, help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--num_tasks", type=int, default=2, help="Number of tasks"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--data_type", type=str, default="mnist", help="Type of data to use"   
    )
    parser.add_argument(
        "--output_size", type=int, default=10, help="Output size of the model"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    
    args = parser.parse_args()
    train(
        seed=args.seed,
        num_tasks=args.num_tasks,
        batch_size=args.batch_size,
        data_type=args.data_type,
        output_size=args.output_size,
        lr=args.lr,
        epochs=args.epochs,
    )

if __name__ == "__main__":
    main()
