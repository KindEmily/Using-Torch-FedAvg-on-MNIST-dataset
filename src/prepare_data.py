# src/prepare_data.py

import pathlib
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_fedavg_assets.dataset.mnist_dataset import setup_mnist
from .organizations import DATA_PROVIDER_ORGS_ID

def prepare_mnist_data():
    # Create the temporary directory for generated data
    (pathlib.Path(__file__).parent.parent / "tmp").mkdir(exist_ok=True)
    data_path = pathlib.Path(__file__).parent.parent / "tmp" / "data_mnist"

    setup_mnist(data_path, len(DATA_PROVIDER_ORGS_ID))

    return data_path

if __name__ == "__main__":
    prepare_mnist_data()