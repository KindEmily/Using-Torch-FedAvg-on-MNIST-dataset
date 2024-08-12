# src/strategy_config.py

from substrafl.strategies import FedAvg
from .metrics import roc_auc, accuracy
from .torch_CNN import TorchCNN
from .data_config import index_generator

def create_strategy(dataset_keys, train_datasample_keys):
    return FedAvg(
        algo=TorchCNN(index_generator),
        metric_functions={"Accuracy": accuracy, "ROC AUC": roc_auc}
    )