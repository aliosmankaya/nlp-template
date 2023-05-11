from dataclasses import dataclass
from torch.cuda import is_available


@dataclass
class CFG:
    model_path: str = "dbmdz/bert-base-turkish-cased"
    max_length: int = 300
    epochs: int = 3
    batch_size: int = 8
    device: str = "cuda" if is_available() else "cpu"
    num_labels: int = 5
    learning_rate: float = 5e-5
