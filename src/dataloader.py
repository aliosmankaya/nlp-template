import pandas as pd
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import CFG


class CustomDataset(Dataset):
    def __init__(self, encodings, target):
        self.encodings = encodings
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        item = {key: tensor(value[idx]) for key, value in self.encodings.items()}
        item["target"] = tensor(self.target[idx])
        return item


def dataloader(data: pd.DataFrame, loader_type: str = "train"):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)

    # Create encodings
    encodings = tokenizer(
        data["text"].values.tolist(),
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
    )

    # Dataset
    dataset = CustomDataset(encodings, data["target"])

    # Dataloader
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=(True if loader_type == "train" else False),
    )
