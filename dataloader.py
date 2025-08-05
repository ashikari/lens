import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

DatasetType = Literal["mnist", "cifar10", "cifar100"]


@dataclass
class DatasetConfig:
    """Configuration for dataset transforms and parameters."""

    input_size: int
    num_channels: int
    num_classes: int


# Dataset configurations
DATASET_CONFIGS = {
    "mnist": DatasetConfig(input_size=28, num_channels=1, num_classes=10),
    "cifar10": DatasetConfig(input_size=32, num_channels=3, num_classes=10),
    "cifar100": DatasetConfig(input_size=32, num_channels=3, num_classes=100),
}


def validate_dataset_name(dataset_name: str) -> DatasetType:
    """Validate and return the dataset name."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(DATASET_CONFIGS.keys())}")
    return dataset_name


def get_transforms(
    dataset_name: DatasetType,
    train: bool = True,
    augment: bool = True,
    custom_transforms: Optional[transforms.Compose] = None,
) -> transforms.Compose:
    """
    Get appropriate transforms for each dataset.

    Args:
        dataset_name: Dataset type
        train: Whether transforms are for training
        augment: Whether to apply data augmentation (only for training)
        custom_transforms: Custom transforms to use instead of default

    Returns:
        Compose transform
    """
    if custom_transforms is not None:
        return custom_transforms

    config = DATASET_CONFIGS[dataset_name]

    # Uniform normalization for all datasets (0.5 mean, 0.5 std for all channels)
    if config.num_channels == 1:
        # MNIST: single channel
        normalize = transforms.Normalize((0.5,), (0.5,))
    else:
        # CIFAR: three channels
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if dataset_name == "mnist":
        if train and augment:
            return transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor(), normalize])
        else:
            return transforms.Compose([transforms.ToTensor(), normalize])
    elif dataset_name in ["cifar10", "cifar100"]:
        if train and augment:
            return transforms.Compose(
                [
                    transforms.RandomCrop(config.input_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            return transforms.Compose([transforms.ToTensor(), normalize])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset_class(dataset_name: DatasetType):
    """Get the appropriate dataset class."""
    dataset_map = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
    }
    return dataset_map[dataset_name]


def get_dataloader(
    dataset_name: DatasetType,
    batch_size: int,
    train: bool,
    num_workers: int = 0,
    labeled: bool = True,
    unlabeled_fraction: float = 0.0,
    data_root: str = "./data",
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
    augment: bool = True,
    custom_transforms: Optional[transforms.Compose] = None,
    seed: int = 42,
) -> DataLoader:
    """
    Generic dataloader function that supports MNIST, CIFAR-10, and CIFAR-100.

    Args:
        dataset_name: Dataset type ("mnist", "cifar10", "cifar100")
        batch_size: Batch size for the dataloader
        train: Whether to use training or validation set
        num_workers: Number of worker processes
        labeled: Whether to return labeled or unlabeled data (only used when unlabeled_fraction > 0)
        unlabeled_fraction: Fraction of data to treat as unlabeled (0.0 to 1.0)
        data_root: Root directory for dataset storage
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        pin_memory: Whether to pin memory for faster GPU transfer
        augment: Whether to apply data augmentation
        custom_transforms: Custom transforms to use
        seed: Random seed for reproducibility

    Returns:
        DataLoader instance
    """
    # Input validation
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"num_workers must be a non-negative integer, got {num_workers}")
    if not 0.0 <= unlabeled_fraction <= 1.0:
        raise ValueError(f"unlabeled_fraction must be between 0.0 and 1.0, got {unlabeled_fraction}")

    # Validate dataset name
    dataset_name = validate_dataset_name(dataset_name)

    # Create data directory if it doesn't exist
    os.makedirs(data_root, exist_ok=True)

    dataset_class = get_dataset_class(dataset_name)
    transform = get_transforms(dataset_name, train, augment, custom_transforms)

    try:
        raw_dataset = dataset_class(
            root=data_root,
            download=True,
            train=train,
            transform=transform,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {str(e)}")

    if unlabeled_fraction > 0:
        torch.manual_seed(seed)
        indices = torch.randperm(len(raw_dataset))

        split = int(unlabeled_fraction * len(indices))

        unlabeled_idx = indices[:split]
        labeled_idx = indices[split:]

        if labeled:
            dataset_subset = Subset(raw_dataset, labeled_idx)
        else:
            dataset_subset = Subset(raw_dataset, unlabeled_idx)
    else:
        dataset_subset = raw_dataset

    return DataLoader(
        dataset=dataset_subset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def get_train_loader(
    batch_size: int,
    num_workers: int,
    dataset_name: DatasetType = "mnist",
    labeled: bool = True,
    unlabeled_fraction: float = 0,
    data_root: str = "./data",
    augment: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Get training dataloader for the specified dataset.

    Args:
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes
        dataset_name: Dataset type ("mnist", "cifar10", "cifar100")
        labeled: Whether to return labeled or unlabeled data (only used when unlabeled_fraction > 0)
        unlabeled_fraction: Fraction of data to treat as unlabeled (0.0 to 1.0)
        data_root: Root directory for dataset storage
        augment: Whether to apply data augmentation
        **kwargs: Additional arguments passed to get_dataloader

    Returns:
        DataLoader instance for training
    """
    if unlabeled_fraction > 0 and not labeled:
        raise ValueError("Cannot request unlabeled data when labeled=False and unlabeled_fraction > 0")

    return get_dataloader(
        dataset_name=dataset_name,
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        labeled=labeled,
        unlabeled_fraction=unlabeled_fraction,
        data_root=data_root,
        augment=augment,
        **kwargs,
    )


def get_validation_loader(
    batch_size: int, num_workers: int, dataset_name: DatasetType = "mnist", data_root: str = "./data", **kwargs
) -> DataLoader:
    """
    Get validation dataloader for the specified dataset.

    Args:
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes
        dataset_name: Dataset type ("mnist", "cifar10", "cifar100")
        data_root: Root directory for dataset storage
        **kwargs: Additional arguments passed to get_dataloader

    Returns:
        DataLoader instance for validation
    """
    return get_dataloader(
        dataset_name=dataset_name,
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        data_root=data_root,
        augment=False,  # No augmentation for validation
        **kwargs,
    )


def get_dataset_info(dataset_name: DatasetType) -> Dict[str, Any]:
    """
    Get information about a dataset.

    Args:
        dataset_name: Dataset type

    Returns:
        Dictionary containing dataset information
    """
    dataset_name = validate_dataset_name(dataset_name)
    config = DATASET_CONFIGS[dataset_name]

    return {
        "name": dataset_name,
        "num_classes": config.num_classes,
        "input_size": config.input_size,
        "num_channels": config.num_channels,
        "normalization": "uniform (0.5, 0.5) for all channels",
    }
