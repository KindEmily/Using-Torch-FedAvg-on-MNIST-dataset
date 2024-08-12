# src/torch_CNN.py

from substrafl.algorithms.pytorch import TorchFedAvgAlgo
from .model import model, criterion, optimizer, seed
from .torch_dataset import TorchDataset

class TorchCNN(TorchFedAvgAlgo):
    def __init__(self, index_generator):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TorchDataset,
            seed=seed,
            use_gpu=False,
        )