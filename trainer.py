from typing import Optional

import torch
from tqdm import tqdm

import wandb
from dataloader import get_train_loader, get_validation_loader
from simple_cnn import CNN_CONFIGS, SimpleCNN


class Trainer:
    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        use_gpu: bool = False,
        logging_interval: int = 100,
        validation_interval: int = 2000,
        max_steps: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.logging_interval = logging_interval
        self.validation_interval = validation_interval
        self.max_steps = max_steps

        self.criterion = torch.nn.CrossEntropyLoss()

        # Check for MPS (Apple Silicon) GPU first, then CUDA GPU
        if use_gpu:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.use_gpu = True
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.use_gpu = True
            else:
                print("Warning: No GPU available (neither MPS nor CUDA). Using CPU instead.")
                self.device = torch.device("cpu")
                self.use_gpu = False
        else:
            self.device = torch.device("cpu")
            self.use_gpu = False

        self.train_loader = get_train_loader(batch_size=self.batch_size, num_workers=0)
        self.validation_loader = get_validation_loader(batch_size=self.batch_size, num_workers=0)

        self.model = SimpleCNN(config=CNN_CONFIGS["nano"])
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        wandb.watch(self.model, log="all", log_freq=self.logging_interval)
        total_steps_per_epoch = len(self.train_loader)
        total_steps = self.num_epochs * total_steps_per_epoch
        global_step = 0

        with tqdm(total=total_steps, desc="Training (all epochs)", unit="it/s", unit_scale=True) as progress_bar:
            for _ in range(self.num_epochs):
                for images, labels in self.train_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    predictions = self.model(images)

                    loss = self.compute_loss(predictions, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if global_step % self.logging_interval == 0:
                        self.log(global_step, loss, progress_bar)

                    if global_step % self.validation_interval == 0:
                        val_loss, val_accuracy = self.validate()
                        self.log_validation(global_step, val_loss, val_accuracy)

                    global_step += 1
                    progress_bar.update(1)

                    if self.max_steps and global_step >= self.max_steps:
                        print("Reached max num steps")
                        return

        # Final validation
        val_loss, val_accuracy = self.validate()
        self.log_validation(global_step, val_loss, val_accuracy, final=True)

    def validate(self):
        """Run validation loop and return validation loss and accuracy."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for images, labels in self.validation_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(images)
                loss = self.compute_loss(predictions, labels)

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(predictions.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        self.model.train()

        avg_loss = total_loss / len(self.validation_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def log(self, step_idx: int, loss: torch.Tensor, progress_bar: tqdm):
        progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})
        wandb.log({"train_loss": loss.item(), "step": step_idx})

    def log_validation(self, step_idx: int, val_loss: float, val_accuracy: float, final: bool = False):
        prefix = "FINAL" if final else f"Step {step_idx}"
        tqdm.write(f"{prefix}: Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
        wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "step": step_idx, "final_validation": final})

    def compute_loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, labels)
