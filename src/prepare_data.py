import pathlib
from torch_fedavg_assets.dataset.mnist_dataset import setup_mnist
from organizations import DATA_PROVIDER_ORGS_ID


# Create the temporary directory for generated data
(pathlib.Path.cwd() / "tmp").mkdir(exist_ok=True)
data_path = pathlib.Path.cwd() / "tmp" / "data_mnist"

setup_mnist(data_path, len(DATA_PROVIDER_ORGS_ID))