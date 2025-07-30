import torchvision
from torch.utils.data import Dataloader


def get_mnist_dataloader(
    batch_size: int, train: bool, num_workers: int = 0
) -> Dataloader:
    dataset = torchvision.datasets.MNIST(
        root="./data",
        download=True,
        train=train,
        transform=torchvision.transforms.ToTensor,
    )
    return Dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        in_order=False,
    )


def get_train_loader(batch_size: int, num_workers: int) -> Dataloader:
    return get_mnist_dataloader(
        train=True, batch_size=batch_size, num_workers=num_workers
    )


def get_validation_loader(batch_size: int, num_workers: int) -> Dataloader:
    return get_mnist_dataloader(
        train=False, batch_size=batch_size, num_workers=num_workers
    )
