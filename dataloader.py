import torch
import torchvision
from torch.utils.data import Dataloader, Subset


def get_mnist_dataloader(
    batch_size: int,
    train: bool,
    num_workers: int = 0,
    labeled=True,
    unlabeled_fraction: float = 0.0,
) -> Dataloader:
    raw_dataset = torchvision.datasets.MNIST(
        root="./data",
        download=True,
        train=train,
        transform=torchvision.transforms.ToTensor,
    )
    if unlabeled_fraction > 0:
        torch.manual_seed(42)
        indices = torch.randperm(len(raw_dataset))

        split = int(unlabeled_fraction * len(indices))

        unlabeled_idx = indices[:split]
        labeled_idx = indices[split:]

        if labeled:
            dataset = Subset(raw_dataset, labeled_idx)
        else:
            dataset = Subset(raw_dataset, unlabeled_idx)

    else:
        dataset = raw_dataset

    return Dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        in_order=False,
    )


def get_train_loader(
    batch_size: int,
    num_workers: int,
    labeled: bool = True,
    unlabeled_fraction: float = 0,
) -> Dataloader:
    msg = f"Invalid Unlabeled Fraction requested. unlabeled_fraction: {unlabeled_fraction}"
    assert not labeled and unlabeled_fraction > 0, msg
    if unlabeled_fraction == 0:
        return get_mnist_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    else:
        # return either the labeled or unlabeled subset of the data
        return get_mnist_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
            labeled=labeled,
            unlabeled_fraction=unlabeled_fraction,
        )


def get_validation_loader(batch_size: int, num_workers: int) -> Dataloader:
    return get_mnist_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
