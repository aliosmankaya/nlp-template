from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold
from torch.cuda import is_available
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall


class Metric:
    auc = AUROC
    accuracy = Accuracy
    f1_score = F1Score
    precision = Precision
    recall = Recall


@dataclass
class CFG:
    task: str = "binary"
    num_labels: int = 2

    model_path: str = "bert-base-uncased"
    epochs: int = 3
    batch_size: int = 8
    max_length: int = 200
    learning_rate: float = 5e-5

    seed: int = 42
    device: str = "cuda" if is_available() else "cpu"

    val_strategy = StratifiedKFold
    n_splits: int = 5
    metric = getattr(Metric, "auc")
