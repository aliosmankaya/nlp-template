import torch
from tqdm import tqdm

from config import CFG


def generic_fn(model, dataloader, metric):
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        target = batch["target"].to(CFG.device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, target=target
        )

        logits = torch.argmax(outputs.logits, dim=1)
        _ = metric(logits, target)
    return metric.compute().item()


def train_fn(model, dataloader, metric):
    model.train()
    generic_fn(model=model, dataloader=dataloader, metric=metric)


def val_fn(model, dataloader, metric):
    model.eval()
    with torch.inference_mode():
        generic_fn(model=model, dataloader=dataloader, metric=metric)
