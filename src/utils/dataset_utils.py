import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


def load_datasets(batch_size: int = 32):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = CIFAR10(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        "./data", train=False, download=True, transform=transform
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


def load_seperate_datasets(num_clients: int, batch_size: int = 32):
    # Data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load full CIFAR-10 datasets
    full_train_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    full_test_dataset = CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Split training set into `num_clients` partitions to simulate different local datasets
    samples_per_client_train = len(full_train_dataset) // num_clients
    samples_per_client_test = len(full_test_dataset) // num_clients

    # Create subsets for each client
    train_dataloaders, test_dataloaders = [], []
    for i in range(num_clients):
        train_start_idx = i * samples_per_client_train
        train_end_idx = (i + 1) * samples_per_client_train
        train_subset = Subset(
            full_train_dataset, list(range(train_start_idx, train_end_idx))
        )
        train_dataloaders.append(
            DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
        )

        test_start_idx = i * samples_per_client_test
        test_end_idx = (i + 1) * samples_per_client_test
        test_subset = Subset(
            full_test_dataset, list(range(test_start_idx, test_end_idx))
        )
        test_dataloaders.append(
            DataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        )

    # Printing the number of samples in each client's dataset
    for i in range(num_clients):
        print(
            f"Client {i+1}:",
            f"Train samples - {len(train_dataloaders[i].dataset)}",
            f"Test samples - {len(test_dataloaders[i].dataset)}",
        )

    return train_dataloaders, test_dataloaders
