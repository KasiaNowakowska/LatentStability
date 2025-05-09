import matplotlib.pyplot as plt
import time
import numpy as np
import wandb

class LossTracker:
    def __init__(self, n_batches_train, n_batches_valid):
        self.n_batches_train = n_batches_train
        self.n_batches_valid = n_batches_valid
        self.best_validation_loss = float("inf")
        self.losses_dict = {}
        self.loss_types = [
            "training_loss",
            "validation_loss",
        ]

        self.init_current_loss_dict()
        self.init_loss_lists_dict()

    def init_current_loss_dict(self):
        self.current_loss_dict = {f"current_{loss_type}": 0 for loss_type in self.loss_types}
        self.losses_dict = self.losses_dict | self.current_loss_dict

    def init_loss_lists_dict(self):
        self.array_losses_dict = {loss_type: [] for loss_type in self.loss_types}
        self.losses_dict = self.losses_dict | self.array_losses_dict

    def reset_current_loss(self):
        for loss_type in self.loss_types:
            self.losses_dict[f"current_{loss_type}"] = 0

    def update_current_loss(self, loss_type, loss_total):
        divider = 1
        if "training" in loss_type:
            divider = self.n_batches_train + 1
        if "validation" in loss_type:
            divider = self.n_batches_valid + 1
        self.losses_dict[f"current_{loss_type}_loss"] += loss_total.numpy(force=True) / divider

    def print_current_loss(self, epoch, loss_type):
        loss_dict = self.losses_dict
        current_loss_key = f"current_{loss_type}_loss"
        print(f"{loss_type}; epoch {epoch}; time {time.time() - self.start_time:.2f}s ")
        print(
            f"total loss {loss_dict[current_loss_key]:.2e}"
        )

    def set_start_time(self, start_time):
        self.start_time = start_time

    def get_losses_dict(self):
        return self.losses_dict

    def get_current_validation_loss(self):
        return self.losses_dict["current_validation_loss"]

    def calculate_and_store_average_losses(self):
        for loss_type in self.loss_types:
            current_loss_key = f"current_{loss_type}"
            average_loss_key = loss_type
            current_loss_value = self.losses_dict[current_loss_key]
            self.losses_dict[average_loss_key].append(current_loss_value)

    def update_wandb_loss(self):
        for loss_type in self.loss_types:
            wandb.log({loss_type: self.losses_dict[loss_type], "epoch": len(self.losses_dict[loss_type])})

    def get_best_validation_loss(self):
        return self.best_validation_loss

    def set_best_validation_loss(self, new_validation_loss):
        self.best_validation_loss = new_validation_loss

    def check_best_validation_loss(self):
        if self.losses_dict["current_validation_loss"] < self.best_validation_loss:
            self.set_best_validation_loss(self.losses_dict["current_validation_loss"])
            return True
        return False

    def create_loss_plot(self, modelpath=None):
        fig, axs = plt.subplots(2, 2, sharex=True)
        loss_type = "training"
        axs[0, 0].plot(self.losses_dict[f"{loss_type}_loss"], label=f"{loss_type}")
        loss_type = "validation"
        axs[0, 0].plot(self.losses_dict[f"{loss_type}_loss"], label=f"{loss_type}")

        axs[0, 0].set_title("Total loss")

        for row in range(2):
            for col in range(2):
                ax = axs[row, col]
                ax.set_xlabel("Epochs")
                ax.set_yscale("log")

        fig.tight_layout()
        if modelpath is not None:
            fig.savefig(modelpath / "loss.png", dpi=100)
            plt.close()
        else:
            plt.show()

    def create_loss_plot2(self, modelpath=None):
        fig, ax = plt.subplots(1, figsize=(8,6))
        loss_type = "training"
        ax.plot(self.losses_dict[f"{loss_type}_loss"], label=f"{loss_type}")
        loss_type = "validation"
        ax.plot(self.losses_dict[f"{loss_type}_loss"], label=f"{loss_type}")

        ax.set_xlabel("Epochs")
        ax.set_yscale("log")
        ax.set_ylabel("MSE_Loss")
        ax.grid()
        ax.legend()

        fig.tight_layout()
        if modelpath is not None:
            fig.savefig(modelpath / "Loss_Plot.png", dpi=100)
            plt.close()
        else:
            plt.show()

    def save_losses(self, path):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "losses.npy", self.losses_dict)
        print(f"--- saved the losses at {path/'losses.npy'} ---")
