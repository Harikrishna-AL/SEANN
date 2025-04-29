import torchvision
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np


# import NMIST data
def get_data(batch_size=128):
    train_data = torchvision.datasets.MNIST(
        root="../../data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.MNIST(
        root="../../data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    # truncate the remaining data that doesn't make a full batch

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, test_loader


# make a function that gives a mnist dataloader that gives a continous data of only classes 0-4 and after that 5-9
def get_MNIST_tasks(batch_size=128, num_tasks=2, max_classes=10):
    train_data = torchvision.datasets.MNIST(
        root="../../data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.MNIST(
        root="../../data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    # train_data_1 = []
    # train_data_2 = []
    # test_data_1 = []
    # test_data_2 = []

    all_train_data = []
    all_test_data = []
    
    per_task = max_classes // num_tasks

    for i in range(per_task, max_classes + 1, per_task):
        task_train_data = []
        task_test_data = []
        for data, target in train_data:
            if i - per_task <= target < i:
                task_train_data.append((data, target))

        for data, target in test_data:
            if i - per_task <= target < i:
                task_test_data.append((data, target))

        train_loader = DataLoader(
            task_train_data, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            task_test_data, batch_size=batch_size, shuffle=False, drop_last=True
        )

        all_train_data.append(train_loader)
        all_test_data.append(test_loader)

    # for data, target in train_data:
    #     if target < 5:
    #         train_data_1.append((data, target))
    #     else:
    #         train_data_2.append((data, target))
    # for data, target in test_data:
    #     if target < 5:
    #         test_data_1.append((data, target))
    #     else:
    #         test_data_2.append((data, target))

    # train_loader_1 = torch.utils.data.DataLoader(
    #     train_data_1, batch_size=batch_size, shuffle=True, drop_last=True
    # )
    # train_loader_2 = torch.utils.data.DataLoader(
    #     train_data_2, batch_size=batch_size, shuffle=True, drop_last=True
    # )
    # test_loader_1 = torch.utils.data.DataLoader(
    #     test_data_1, batch_size=batch_size, shuffle=False, drop_last=True
    # )
    # test_loader_2 = torch.utils.data.DataLoader(
    #     test_data_2, batch_size=batch_size, shuffle=False, drop_last=True
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=batch_size, shuffle=False, drop_last=True
    # )

    # return train_loader_1, train_loader_2, test_loader_1, test_loader_2, test_loader
    # unroll the list of dataloaders and return them
    return all_train_data, all_test_data


def get_domain_inc_data(batch_size=128):
    train_data = torchvision.datasets.MNIST(
        root="../../data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.MNIST(
        root="../../data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),
            torchvision.transforms.RandomAffine(0, shear=10),
            torchvision.transforms.RandomAffine(0, scale=(0.8, 1.2)),
            # add random noise
            torchvision.transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
        ]
    )

    transformed_train_data = torchvision.datasets.MNIST(
        root="../../data",
        train=True,
        download=True,
        transform=transforms,
    )

    transformed_test_data = torchvision.datasets.MNIST(
        root="../../data",
        train=False,
        download=True,
        transform=transforms,
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, drop_last=True
    )

    transformed_train_loader = DataLoader(
        transformed_train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    transformed_test_loader = DataLoader(
        transformed_test_data, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return train_loader, transformed_train_loader, test_loader, transformed_test_loader



def get_EMNIST_tasks(batch_size=128, num_tasks=2, max_classes=26):

    # EMNIST letters: 0-25=A-Z, 26-51=a-z

    # task_letter_groups = [
    #     [0, 3, 14, 16],     # Task 1: A, D, O, Q (round shapes)
    #     [15, 17, 1],        # Task 2: P, R, B (curves + lines)
    #     [2, 6, 20],         # Task 3: C, G, U (open curves)
    #     [9, 24, 11],        # Task 4: J, Y, L (hooks/diagonals)
    #     [4, 5, 7, 8],       # Task 5: E, F, H, I (horizontal/vertical lines)
    #     [19, 25, 10],       # Task 6: T, Z, K (crossbars/diagonals)
    #     [12, 13, 22, 21],   # Task 7: M, N, W, V (slanted lines)
    #     [23, 18],           # Task 8: X, S (crossed/curves)
    #     [26, 27, 29, 30],   # Task 9: a, b, d, e (lowercase mixed)
    #     [32, 42, 41]        # Task 10: g, q, p (lowercase loops)
    # ][:num_tasks]  # Slice if fewer tasks needed

    # max_task_letter_groups = []

    # # choose letter groups for the specified number of tasks
    # for i in range(num_tasks):
    #     # Get the letters for the current task
    #     max_task_letter_groups.append(task_letter_groups[i])
        

    # Load full dataset
    train_data = torchvision.datasets.EMNIST(
        root="../../data",
        split="letters",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.EMNIST(
        root="../../data",
        split="letters",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    all_train_data = []
    all_test_data = []
    
    per_task = max_classes // num_tasks # 13

    for i in range(per_task, max_classes + 1, per_task):
        task_train_data = []
        task_test_data = []
        for data, target in train_data:
            target = target - 1  # Adjust target to be 0-indexed
            if i - per_task <= target < i:
                task_train_data.append((data, target))

        for data, target in test_data:
            target = target - 1  # Adjust target to be 0-indexed
            if i - per_task <= target < i:
                task_test_data.append((data, target))

        train_loader = DataLoader(
            task_train_data, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            task_test_data, batch_size=batch_size, shuffle=False, drop_last=True
        )

        all_train_data.append(train_loader)
        all_test_data.append(test_loader)


    return (all_train_data, all_test_data)



def get_FashionMNIST_tasks(batch_size=128, num_tasks=10, max_classes=10):
    """Returns task loaders with grouped Fashion MNIST classes for automatic overlap.
       Format: train_loader_1, ..., train_loader_N, 
               test_loader_1, ..., test_loader_N,
               test_loader_all"""
    
    # Fashion MNIST class groupings by visual similarity
    # task_class_groups = [
    #     [0, 6],        # Task 1: T-shirt/top + Shirt (top clothing)
    #     [1, 3, 5],      # Task 2: Trouser + Dress + Sandal (bottom/flowy)
    #     [2, 4, 8],      # Task 3: Pullover + Coat + Bag (upper body/accessories)
    #     [7, 9],         # Task 4: Sneaker + Ankle boot (footwear)
    #     [0, 2, 6, 8],   # Task 5: T-shirt + Pullover + Shirt + Bag (mixed tops)
    #     [1, 3, 5, 7],   # Task 6: Trouser + Dress + Sandal + Sneaker (mixed bottom)
    #     [4, 9],         # Task 7: Coat + Ankle boot (outerwear)
    #     [2, 4],         # Task 8: Pullover + Coat (warm tops)
    #     [5, 7, 9],      # Task 9: Sandal + Sneaker + Ankle boot (all footwear)
    #     [0, 1, 2, 3]    # Task 10: T-shirt + Trouser + Pullover + Dress (core clothing)
    # ][:num_tasks]  # Limit to requested tasks


    # max_task_class_groups = []

    # # choose letter groups for the specified number of tasks
    # for i in range(num_tasks):
    #     # Get the letters for the current task
    #     max_task_class_groups.append(task_class_groups[i])
        

    # Load full dataset
    train_data = torchvision.datasets.FashionMNIST(
        root="../../data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.FashionMNIST(
        root="../../data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    all_train_data = []
    all_test_data = []
    
    per_task = max_classes // num_tasks

    for i in range(per_task, max_classes + 1, per_task):
        task_train_data = []
        task_test_data = []
        for data, target in train_data:
            if i - per_task <= target < i:
                task_train_data.append((data, target))

        for data, target in test_data:
            if i - per_task <= target < i:
                task_test_data.append((data, target))

        train_loader = DataLoader(
            task_train_data, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            task_test_data, batch_size=batch_size, shuffle=False, drop_last=True
        )

        all_train_data.append(train_loader)
        all_test_data.append(test_loader)

    # test_loader = DataLoader(
    #     test_data, batch_size=batch_size, shuffle=False, drop_last=True
    # )

    return (all_train_data, all_test_data)