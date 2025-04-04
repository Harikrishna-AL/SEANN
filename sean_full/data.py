import torchvision
import torch


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

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, test_loader


# make a function that gives a mnist dataloader that gives a continous data of only classes 0-4 and after that 5-9
def get_data_separate(batch_size=128):
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
        [   torchvision.transforms.ToTensor(),
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

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, drop_last=True
    )

    transformed_train_loader = torch.utils.data.DataLoader(
        transformed_train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    transformed_test_loader = torch.utils.data.DataLoader(
        transformed_test_data, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return train_loader, transformed_train_loader, test_loader, transformed_test_loader