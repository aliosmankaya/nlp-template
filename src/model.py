from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from src.config import CFG


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_path = CFG.model_path
        self.config = AutoConfig.from_pretrained(
            self.model_path, num_labels=CFG.num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            config=self.config,
        )

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
