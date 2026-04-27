import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

SAVE_DIR = Path(__file__).resolve().parents[2] / "trained_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


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
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: str,
        best_model_filename: str,
        accumulation_steps: int = 1,
        num_epochs: int = 50,
        config: dict = None,
        training_logs: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.best_model_filename = best_model_filename
        self.num_epochs = num_epochs
        self.best_val_loss = float("inf")
        self.training_logs = training_logs
        self.scaler = torch.amp.GradScaler(self.device)

        # --- LOGGING SETUP ---
        self.log_filename = best_model_filename.replace(".pth", ".txt")
        self.log_filepath = SAVE_DIR / self.log_filename

        # Initial header in the log file
        self._log("=" * 32)
        self._log("--- STARTING TRAINING ENGINE ---")
        self._log("=" * 32)

        if config:
            self._log("--- HYPERPARAMETERS ---")
            for key, value in config.items():
                self._log(f"{key}: {value}")
            self._log("=" * 32 + "\n")

    def _log(self, message: str):
        """Prints a message to the console and appends it to the log file."""
        print(message)
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def train(self):
        for epoch in range(self.num_epochs):
            if self.training_logs:
                self._log(f"-- Epoch: {epoch + 1}/{self.num_epochs} --")

            start_time = time.time()

            self.model.train()
            train_loss = 0.0

            self.optimizer.zero_grad(set_to_none=True)

            for i, batch in enumerate(self.train_loader):
                if self.training_logs:
                    batch_start_time = time.time()

                inputs = batch["image"].to(self.device, non_blocking=True)
                targets = batch["label"].float().to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                unscaled_loss = loss.item()
                loss = loss / self.accumulation_steps

                self.scaler.scale(loss).backward()
                if ((i + 1) % self.accumulation_steps == 0) or (
                    (i + 1) == len(self.train_loader)
                ):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                train_loss += unscaled_loss

                if self.training_logs:
                    batch_end_time = time.time()
                    batch_duration = batch_end_time - batch_start_time
                    self._log(
                        f"Training batch {i+1}/{len(self.train_loader)}: "
                        f"{int(batch_duration//60)}m {int(batch_duration%60)}s {int((batch_duration*1000)%1000)}ms"
                    )

            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for i, batch in enumerate(self.val_loader):
                    if self.training_logs:
                        batch_start_time = time.time()

                    inputs = batch["image"].to(self.device, non_blocking=True)
                    targets = batch["label"].float().to(self.device, non_blocking=True)

                    with torch.autocast(device_type=self.device, dtype=torch.float16):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)

                    val_loss += loss.item()

                    if self.training_logs:
                        batch_end_time = time.time()
                        batch_duration = batch_end_time - batch_start_time
                        self._log(
                            f"Validation batch {i+1}/{len(self.val_loader)}: "
                            f"{int(batch_duration//60)}m {int(batch_duration%60)}s {int((batch_duration*1000)%1000)}ms"
                        )

            end_time = time.time()
            epoch_duration = end_time - start_time
            epoch_mins, epoch_secs = divmod(epoch_duration, 60)

            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            self._log(f"----- Epoch: {epoch + 1}/{self.num_epochs} ------")
            self._log(f"Time            : {int(epoch_mins)}m {int(epoch_secs)}s")
            self._log(f"Learning Rate   : {current_lr:.6f}")
            self._log(f"Training Loss   : {avg_train_loss:.4f}")
            self._log(f"Validation Loss : {avg_val_loss:.4f}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                best_model_path = SAVE_DIR / self.best_model_filename
                torch.save(self.model.state_dict(), best_model_path)
                self._log("Best model saved!")

            self.scheduler.step(avg_val_loss)

            self._log("------------------------")

        self._log("Training Completed Successfully!")
