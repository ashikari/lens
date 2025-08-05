import argparse
from typing import Optional

import wandb

from simple_cnn import CNN_CONFIGS
from trainer import Trainer

wandb.login()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default=None, help="Optional run name for W&B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument(
        "--model",
        type=str,
        default=list(CNN_CONFIGS.keys())[0],
        choices=list(CNN_CONFIGS.keys()),
        help="Which CNN model configuration to use",
    )
    args = parser.parse_args()
    return args


def get_unique_run_name(project: str, base_name: Optional[str] = None):
    """Generate a unique run name for W&B given a project and optional base name."""
    if base_name is None:
        return None

    api = wandb.Api()
    # Use the current logged-in entity (user or team) for the project
    entity = wandb.run.entity if wandb.run is not None else None
    if entity is None:
        # Try to get from environment variable or fallback to None
        import os

        entity = os.environ.get("WANDB_ENTITY", None)
    runs = api.runs(f"{entity}/{project}" if entity else project)

    existing_names = {run.name for run in runs if run.name.startswith(base_name)}
    if base_name not in existing_names:
        return base_name

    i = 1
    while f"{base_name}_{i}" in existing_names:
        i += 1
    return f"{base_name}_{i}"


if __name__ == "__main__":
    args = parse_args()

    wandb_project = "mnist"
    name = get_unique_run_name(wandb_project, args.name)

    with wandb.init(
        name=name,
        project=wandb_project,
        config=args,
        mode="disabled" if args.disable_wandb else "online",
    ):
        config = wandb.config
        trainer = Trainer(batch_size=config.batch_size, num_epochs=config.num_epochs, use_gpu=args.use_gpu)
        trainer.train()
