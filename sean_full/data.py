import torchvision
import torch
from torchvision import transforms


# import NMIST data
def get_data(batch_size=128):
    train_data = torchvision.datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    # truncate the remaining data that doesn't make a full batch

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, test_loader


# make a function that gives a mnist dataloader that gives a continous data of only classes 0-4 and after that 5-9
def get_data_separate(batch_size=128, type="mnist"):
    if type=="mnist":
        train_data = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        test_data = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif type == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
            
            ]
        )
        train_data = torchvision.datasets.CIFAR10(
            root="../data",
            train=True,
            download=True,
            transform=transform,
        )
        test_data = torchvision.datasets.CIFAR10(
            root="../data",
            train=False,
            download=True,
            transform=transform,
        )
        

    train_data_1 = []
    train_data_2 = []
    test_data_1 = []
    test_data_2 = []
    for data, target in train_data:

        if target < 5:
            train_data_1.append((data, target))
        else:
            train_data_2.append((data, target))
    for data, target in test_data:
        if target < 5:
            test_data_1.append((data, target))
        else:
            test_data_2.append((data, target))

    train_loader_1 = torch.utils.data.DataLoader(
        train_data_1, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader_2 = torch.utils.data.DataLoader(
        train_data_2, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader_1 = torch.utils.data.DataLoader(
        test_data_1, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_2 = torch.utils.data.DataLoader(
        test_data_2, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return train_loader_1, train_loader_2, test_loader_1, test_loader_2


def get_data_separate_cifar_10(batch_size=128):
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale(num_output_channels=1)]
    )
    train_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms,
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms,
    )

    train_data_1 = []
    train_data_2 = []
    train_data_3 = []
    train_data_4 = []
    train_data_5 = []
    test_data_1 = []
    test_data_2 = []
    test_data_3 = []
    test_data_4 = []
    test_data_5 = []

    train_loader_list = []
    test_loader_list = []



    # separate the data into 2 classes each based on cifar10 labels
    for data, target in train_data:
        if target == 0 or target == 1:
            train_data_1.append((data, target))
        elif target == 2 or target == 3:
            train_data_2.append((data, target))
        elif target == 4 or target == 5:
            train_data_3.append((data, target))
        elif target == 6 or target == 8:
            train_data_4.append((data, target)) 
        else:
            train_data_5.append((data, target))

    for data, target in test_data:
        if target == 0 or target == 1:
            test_data_1.append((data, target))
        elif target == 2 or target == 3:
            test_data_2.append((data, target))
        elif target == 4 or target == 5:
            test_data_3.append((data, target))
        elif target == 6 or target == 8:
            test_data_4.append((data, target))
        else:
            test_data_5.append((data, target))

    train_loader_1 = torch.utils.data.DataLoader(
        train_data_1, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader_list.append(train_loader_1)

    train_loader_2 = torch.utils.data.DataLoader(
        train_data_2, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader_list.append(train_loader_2)

    train_loader_3 = torch.utils.data.DataLoader(
        train_data_3, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader_list.append(train_loader_3)

    train_loader_4 = torch.utils.data.DataLoader(
        train_data_4, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader_list.append(train_loader_4)

    train_loader_5 = torch.utils.data.DataLoader(
        train_data_5, batch_size=batch_size, shuffle=True, drop_last=True
    )
    train_loader_list.append(train_loader_5)

    test_loader_1 = torch.utils.data.DataLoader(
        test_data_1, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_list.append(test_loader_1)

    test_loader_2 = torch.utils.data.DataLoader(
        test_data_2, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_list.append(test_loader_2)

    test_loader_3 = torch.utils.data.DataLoader(
        test_data_3, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_list.append(test_loader_3)

    test_loader_4 = torch.utils.data.DataLoader(
        test_data_4, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_list.append(test_loader_4)

    test_loader_5 = torch.utils.data.DataLoader(
        test_data_5, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader_list.append(test_loader_5)

    return train_loader_list, train_loader_list



def get_data_separate_dynamic(label_groups, batch_size=128):
    """
    Create dataloaders for CIFAR-10 data split into dynamic label groups.

    Args:
        label_groups (list of list of int): List where each sublist contains the labels for one group.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        tuple: Two lists - train dataloaders and test dataloaders.
    """
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale(num_output_channels=1)]
    )
    train_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms,
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms,
    )

    # Create dictionaries to store data for each group
    train_data_groups = {i: [] for i in range(len(label_groups))}
    test_data_groups = {i: [] for i in range(len(label_groups))}

    # Separate train data into groups
    for data, target in train_data:
        for group_idx, labels in enumerate(label_groups):
            if target in labels:
                train_data_groups[group_idx].append((data, target))
                break

    # Separate test data into groups
    for data, target in test_data:
        for group_idx, labels in enumerate(label_groups):
            if target in labels:
                test_data_groups[group_idx].append((data, target))
                break

    # Create dataloaders for each group
    train_loader_list = []
    test_loader_list = []

    for group_idx in range(len(label_groups)):
        train_loader = torch.utils.data.DataLoader(
            train_data_groups[group_idx], batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data_groups[group_idx], batch_size=batch_size, shuffle=False, drop_last=True
        )

        train_loader_list.append(train_loader)
        test_loader_list.append(test_loader)

    return train_loader_list, test_loader_list