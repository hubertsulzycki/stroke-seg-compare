import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Universal Training Engine for 2D, 2.5D, and 3D PyTorch models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        num_epochs: int = 50,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0

            for batch in self.train_loader:
                inputs = batch["image"].to(self.device)
                targets = batch["label"].float().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                self.model.eval()

            with torch.no_grad():
                val_loss = 0

                for batch in self.val_loader:
                    inputs = batch["image"].to(self.device)
                    targets = batch["label"].float().to(self.device)

                    outputs = self.model(inputs)

                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()

                    self.model.eval()

            print(f"--- Epoch: {epoch} ---")
            print(f"Training loss : {train_loss / len(self.train_loader)}")
            print(f"Validation loss: {val_loss / len(self.val_loader)}")
            print("----------------------")
