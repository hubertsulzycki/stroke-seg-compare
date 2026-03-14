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
            train_loss = 0.0

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

            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)

            print(f"--- Epoch: {epoch+1}/{self.num_epochs} ---")
            print(f"Training loss : {avg_train_loss}")
            print(f"Validation loss: {avg_val_loss}")
            print("----------------------")
