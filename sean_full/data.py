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

def get_cifar10_data(batch_size=128, max_classes=10, num_tasks=2):
    train_data = torchvision.datasets.CIFAR10(
        root="../../data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.CIFAR10(
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

        train_loader = torch.utils.data.DataLoader(
            task_train_data, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            task_test_data, batch_size=batch_size, shuffle=False, drop_last=True
        )

        all_train_data.append(train_loader)
        all_test_data.append(test_loader)
        
    return all_train_data, all_test_data
    

    
# make a function that gives a mnist dataloader that gives a continous data of only classes 0-4 and after that 5-9
def get_data_separate(batch_size=128, num_tasks=2, max_classes=10):
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

        train_loader = torch.utils.data.DataLoader(
            task_train_data, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            task_test_data, batch_size=batch_size, shuffle=False, drop_last=True
        )

        all_train_data.append(train_loader)
        all_test_data.append(test_loader)

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
