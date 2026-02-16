import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass


def weighted_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Weighted Pearson Correlation Coefficient.

    This metric emphasizes performance on data points with larger target amplitudes
    (larger price movements) by using the absolute value of the target as a sample weight.

    Predictions are clipped to the range [-6, 6] before calculation to prevent
    outliers from dominating the metric.

    Args:
        y_true: Ground truth target values (numpy array).
        y_pred: Predicted values (numpy array).

    Returns:
        float: Weighted Pearson correlation coefficient.
    """
    # Clip predictions to valid range [-6, 6]
    y_pred_clipped = np.clip(y_pred, -6.0, 6.0)

    # Calculate weights based on target amplitude
    weights = np.abs(y_true)
    weights = np.maximum(weights, 1e-8)

    # Calculate weighted means
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 0.0

    mean_true = np.sum(y_true * weights) / sum_w
    mean_pred = np.sum(y_pred_clipped * weights) / sum_w

    # Calculate weighted deviations
    dev_true = y_true - mean_true
    dev_pred = y_pred_clipped - mean_pred

    # Calculate weighted covariance
    cov = np.sum(weights * dev_true * dev_pred) / sum_w

    # Calculate weighted variances
    var_true = np.sum(weights * dev_true**2) / sum_w
    var_pred = np.sum(weights * dev_pred**2) / sum_w

    # Compute correlation
    if var_true <= 0 or var_pred <= 0:
        return 0.0

    corr = cov / (np.sqrt(var_true) * np.sqrt(var_pred))

    return float(corr)


@dataclass
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    #
    state: np.ndarray


class PredictionModel:
    def predict(self, data_point: DataPoint) -> np.ndarray:
        # return dummy prediction
        return np.zeros(2)


class ScorerStepByStep:
    def __init__(self, dataset_path: str):
        self.dataset = pd.read_parquet(dataset_path)

        # Calc feature dimension: first 3 columns are seq_ix, step_in_seq & need_prediction
        # Total columns: 3 metadata + 32 features + 2 targets = 37
        # Features are cols [3:35]
        self.dim = 2
        self.features = self.dataset.columns[3:35]
        self.targets = self.dataset.columns[35:]

    def score(self, model: PredictionModel) -> dict:
        predictions = []
        targets = []

        prediction = None

        # Iterate over numpy array for speed
        for row in tqdm(self.dataset.values):
            seq_ix = row[0]
            step_in_seq = row[1]
            need_prediction = row[2]
            lob_data = row[2:35]  # 33 features
            labels = row[35:]  # 2 targets
            #
            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, lob_data)
            prediction = model.predict(data_point)

            self.check_prediction(data_point, prediction)
            if prediction is not None:
                predictions.append(prediction)
                targets.append(labels)

        # report metrics
        return self.calc_metrics(np.array(predictions), np.array(targets))

    def check_prediction(self, data_point: DataPoint, prediction: np.ndarray):
        if not data_point.need_prediction:
            if prediction is not None:
                raise ValueError(f"Prediction is not needed for {data_point}")
            return

        if prediction is None:
            raise ValueError(f"Prediction is required for {data_point}")

        if prediction.shape[0] != self.dim:
            raise ValueError(
                f"Prediction has wrong shape: {prediction.shape[0]} != {self.dim}"
            )

    def calc_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        scores = {}
        for ix_target, target_name in enumerate(self.targets):
            scores[target_name] = weighted_pearson_correlation(
                targets[:, ix_target], predictions[:, ix_target]
            )
        scores["weighted_pearson"] = np.mean(list(scores.values()))
        return scores
