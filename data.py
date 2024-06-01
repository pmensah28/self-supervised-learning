import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    cifar10_train = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10('../data', train=False, download=True, transform=transform)

    unsupervised_pretrain, supervised_train = random_split(cifar10_train, [45000, 5000])

    train_loader_unsupervised = DataLoader(unsupervised_pretrain, batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader_supervised = DataLoader(supervised_train, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader_unsupervised, train_loader_supervised, test_loader
