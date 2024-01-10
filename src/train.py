import argparse

from config import CFG, Metric
from dataloader import dataloader
from model import CustomModel
from preprocess import preprocess_fn
from utils import train_fn, val_fn


class Train:
    def __init__(self, data_path: str, metric: str = None):
        self.data_path = data_path
        self.val_strategy = CFG.val_strategy(
            n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed
        )
        self.metric = CFG.metric(task=CFG.task, num_labels=CFG.num_labels).to(
            CFG.device
        )
        if metric:
            self.metric = getattr(Metric, metric)(
                task=CFG.task, num_labels=CFG.num_labels
            ).to(CFG.device)

    def _data(self):
        return preprocess_fn(self.data_path)

    def run(self):
        data = self._data()
        print(CFG.device)
        for fold, (train_index, test_index) in enumerate(
            self.val_strategy.split(data["text"], data["labels"])
        ):
            train = data.loc[train_index].reset_index().drop("index", axis=1)
            test = data.loc[test_index].reset_index().drop("index", axis=1)

            train_dataloader = dataloader(data=train, loader_type="train")
            test_dataloader = dataloader(data=test, loader_type="test")

            model = CustomModel().to(CFG.device)

            print("\n")
            print("\n")
            print("Fold", fold + 1)
            for epoch in range(3):
                print("\n")
                print("Epoch", epoch)

                print("Train step")
                train_fn(model=model, dataloader=train_dataloader, metric=self.metric)

                print("Val step")
                val_fn(model=model, dataloader=test_dataloader, metric=self.metric)

                print("Score:", self.metric.compute().item())

                self.metric.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="train data path")
    parser.add_argument("--metric", help="evaluation metric", default="auc")
    args = parser.parse_args()

    train = Train(data_path=args.path)
    train.run()
