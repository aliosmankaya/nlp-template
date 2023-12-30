from src.config import CFG
from src.dataloader import dataloader
from src.model import CustomModel
from src.preprocess import preprocess_fn
from src.utils import train_fn, val_fn


class Train:
    def __init__(self, data_path):
        self.data_path = data_path
        self.val_strategy = CFG.val_strategy(
            n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed
        )
        self.metric = CFG.metric(task=CFG.task, num_labels=CFG.num_labels)

    def _data(self):
        return preprocess_fn(self.data_path)

    def run(self):
        data = self._data()

        for fold, (train_index, test_index) in enumerate(
            self.val_strategy.split(data["text"], data["target"])
        ):
            print("Fold", fold + 1)
            train = data.loc[train_index].reset_index().drop("index", axis=1)
            test = data.loc[test_index].reset_index().drop("index", axis=1)

            train_dataloader = dataloader(data=train, loader_type="train")
            test_dataloader = dataloader(data=test, loader_type="test")

            model = CustomModel().to(CFG.device)

            for epoch in range(3):
                print("\n")
                print("Epoch", epoch)

                print("Train step")
                train_fn(model=model, dataloader=train_dataloader, metric=self.metric)

                print("Val step")
                val_fn(model=model, dataloader=test_dataloader, metric=self.metric)

                print("\n")
                print("F1 Score:", self.metric.compute().item())

                self.metric.reset()
