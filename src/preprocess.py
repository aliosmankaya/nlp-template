import pandas as pd


def preprocess_fn(data_path: str):
    data = pd.read_csv(data_path)
    """
    Task specific preprocess could be here
    """
    return data
