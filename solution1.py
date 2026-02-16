import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import lightning as L
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Metric
from lightning.pytorch.callbacks import RichProgressBar, EarlyStopping


class WeightedPearsonCorr(Metric):
    def __init__(self):
        super().__init__()
        # add_state ensures these are moved to the correct device (CPU/GPU)
        # and reset automatically between epochs
        self.add_state("sum_w", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_wy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_wx", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_wyy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_wxx", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_wxy", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: Predicted values [Batch, Seq]
            target: Ground truth values [Batch, Seq]
        """

        # 2. Clip and calculate weights (Only on valid points)
        preds = torch.clamp(preds, -6.0, 6.0)
        weights = torch.abs(target)
        weights = torch.clamp(weights, min=1e-8)

        # 3. Update running sums
        self.sum_w += torch.sum(weights)
        self.sum_wy += torch.sum(weights * target)
        self.sum_wx += torch.sum(weights * preds)
        self.sum_wyy += torch.sum(weights * target * target)
        self.sum_wxx += torch.sum(weights * preds * preds)
        self.sum_wxy += torch.sum(weights * target * preds)

    def compute(self):
        # 3. Final calculation using the aggregated sums
        if self.sum_w == 0:
            return torch.tensor(0.0)

        mean_y = self.sum_wy / self.sum_w
        mean_x = self.sum_wx / self.sum_w

        # Variance/Covariance formulas using Expected Values:
        # Var(X) = E[X^2] - (E[X])^2
        var_y = (self.sum_wyy / self.sum_w) - (mean_y**2)
        var_x = (self.sum_wxx / self.sum_w) - (mean_x**2)
        cov = (self.sum_wxy / self.sum_w) - (mean_y * mean_x)

        if var_y <= 0 or var_x <= 0:
            return torch.tensor(0.0)

        return cov / (torch.sqrt(var_y) * torch.sqrt(var_x))


class SequenceDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_cols: list,
        target_cols: list,
        num_steps: int,
    ):
        # 1. Sort to ensure sequences and steps are in the correct order
        # Assuming 'step' is the column name for the sequence order
        df_sorted = dataframe.sort_values(["seq_ix", "step_in_seq"])

        num_seqs = df_sorted["seq_ix"].nunique()

        # 2. Extract and Reshape X (Features)
        # Shape: [Num_Seqs, 1000, 33]
        x_raw = df_sorted[feature_cols].values.astype("float32")
        self.X = torch.from_numpy(x_raw.reshape(num_seqs, num_steps, len(feature_cols)))

        # 3. Extract and Reshape Y (Targets)
        # Shape: [Num_Seqs, 1000, 2]
        y_raw = df_sorted[target_cols].values.astype("float32")
        self.Y = torch.from_numpy(y_raw.reshape(num_seqs, num_steps, len(target_cols)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Returns a tuple of (features, targets) for one sequence
        return self.X[idx], self.Y[idx]


class SequenceModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        output_size=2,
        use_need_pred_for_training=False,
    ):
        super(SequenceModel, self).__init__()

        # The LSTM processes the sequence
        # batch_first=True means we provide data as [Batch, Seq, Feature]
        self.lstm = nn.LSTM(
            input_size=input_size if use_need_pred_for_training else input_size - 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,  # Adds robustness
        )

        # Fully connected layer to map hidden state to your 2 targets
        self.fc = nn.Linear(hidden_size, output_size)
        self.use_need_pred_for_training = use_need_pred_for_training

    def forward(self, x):
        # x shape: [Batch, num_sequence_per_step, num_features]

        # out shape: [Batch, num_sequence_per_step, num_targets]

        # _ contains the hidden and cell states (we don't need them for many-to-many)
        if self.use_need_pred_for_training:
            out, _ = self.lstm(x)
        else:
            out, _ = self.lstm(x[:, :, 1:])

        # We pass every timestep through the linear layer
        # result shape: [Batch, 1000, 2]
        prediction = self.fc(out)

        return prediction


# define the LightningModule
class SeqLitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.val_corr = WeightedPearsonCorr()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        x, t = batch
        # skip need_predictions
        t_hat = self.model(x)
        loss = nn.functional.mse_loss(t_hat, t)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)  # Output shape [Batch, Seq, num_outputs]
        # print(f"Shape of y_hat: {y_hat.shape}")

        # Extract the 'need_prediction' flag from the first feature of every step
        # Shape becomes [Batch, Seq]
        mask = x[:, :, 0].bool()

        # timepass[mask.bool()].reshape(timepass.shape[0], -1)
        preds = y_hat[mask].reshape(y_hat.shape[0], -1)
        target = y[mask].reshape(y.shape[0], -1)

        # Update the metric with the predictions, targets, and the mask
        # We use .squeeze(-1) on y and y_hat to ensure they match the mask shape
        self.val_corr.update(preds, target)
        mse = F.mse_loss(preds, target)

        self.log(
            "val_weighted_corr",
            self.val_corr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val_mse", mse, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def create_dataloader(df, features, targets, batch_size, num_steps_per_seq):
    dataset = SequenceDataset(df, features, targets, num_steps_per_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def main():
    BATCH_SIZE = 128
    HIDDEN_SIZE = 128
    NUM_HIDDEN_LAYERS = 4
    LIMIT_TRAIN_BATCHES = 1.0  # LIMIT_TRAIN_BATCHES = 10
    NUM_EPOCHS = 20  # NUM_EPOCHS = 1
    NUM_STEPS_PER_SEQ = 1000

    gpu_or_cpu = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using {gpu_or_cpu} as device.")

    need_predictions_feature = ["need_prediction"]
    bid_price_features = [f"p{i}" for i in range(6)]
    ask_price_features = [f"p{i}" for i in range(6, 12)]
    bid_volume_features = [f"v{i}" for i in range(6)]
    ask_volume_features = [f"v{i}" for i in range(6, 12)]
    trade_price_features = [f"dp{i}" for i in range(4)]
    trade_volume_features = [f"dv{i}" for i in range(4)]
    features = (
        need_predictions_feature
        + bid_price_features
        + ask_price_features
        + bid_volume_features
        + ask_volume_features
        + trade_volume_features
        + trade_price_features
    )

    targets = ["t0", "t1"]

    print(f"Features: {features}")
    print(f"Targets: {targets}")

    train_df = pd.read_parquet("datasets/train.parquet")
    print("Train DataFrame shape:", train_df.shape)
    valid_df = pd.read_parquet("datasets/valid.parquet")
    print("Valid DataFrame shape:", valid_df.shape)

    train_loader = create_dataloader(
        train_df,
        features,
        targets,
        batch_size=BATCH_SIZE,
        num_steps_per_seq=NUM_STEPS_PER_SEQ,
    )
    valid_loader = create_dataloader(
        valid_df,
        features,
        targets,
        batch_size=BATCH_SIZE,
        num_steps_per_seq=NUM_STEPS_PER_SEQ,
    )

    print(f"Train DataLoader created with batch size {BATCH_SIZE}.")

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    model = SequenceModel(
        input_size=len(features),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_HIDDEN_LAYERS,
        output_size=len(targets),
    )

    lit_model = SeqLitModel(model)
    early_stop_callback = EarlyStopping(
        monitor="val_weighted_corr",  # The string used in self.log()
        min_delta=0.00,  # Minimum change to qualify as an improvement
        patience=3,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode="max",  # "max" for correlation, "min" for loss
    )
    trainer = L.Trainer(
        accelerator=gpu_or_cpu,
        limit_train_batches=LIMIT_TRAIN_BATCHES,
        max_epochs=NUM_EPOCHS,
        callbacks=[RichProgressBar(), early_stop_callback],
    )
    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
    # trainer.test(ckpt_path="best")
    # trainer.test(ckpt_path="last")
    # trainer.evaluate(model=lit_model, dataloaders=valid_loader)


if __name__ == "__main__":
    main()
