# Help needed🥲
Please help me to understand how to fix the issue with  
```
ModuleNotFoundError: No module named 'src'
```

# Implemented solution: 
Instead of using modular project structure - put everything inside a single flat structure 
The final code looks like this: 

```
# src/main.py

# Setup

from substra import Client

N_CLIENTS = 3

client_0 = Client(client_name="org-1")
client_1 = Client(client_name="org-2")
client_2 = Client(client_name="org-3")

# Create a dictionary to easily access each client from its human-friendly id
clients = {
    client_0.organization_info().organization_id: client_0,
    client_1.organization_info().organization_id: client_1,
    client_2.organization_info().organization_id: client_2,
}

# Store organization IDs
ORGS_ID = list(clients)
ALGO_ORG_ID = ORGS_ID[0]  # Algo provider is defined as the first organization.
DATA_PROVIDER_ORGS_ID = ORGS_ID[1:]  # Data providers orgs are the two last organizations.



# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Number of organizations: {len(ORGS_ID)}")
logger.info(f"Algorithm provider: {ALGO_ORG_ID}")
logger.info(f"Data providers: {DATA_PROVIDER_ORGS_ID}")







# Data and metrics
## Data preparation

import pathlib
from torch_fedavg_assets.dataset.mnist_dataset import setup_mnist


# Create the temporary directory for generated data
(pathlib.Path.cwd() / "tmp").mkdir(exist_ok=True)
data_path = pathlib.Path.cwd() / "tmp" / "data_mnist"

setup_mnist(data_path, len(DATA_PROVIDER_ORGS_ID))


print(f"DATA PATH <#################################################3 {data_path}")
print(f"DATA PATH <#################################################3 {data_path}")
print(f"DATA PATH <#################################################3 {data_path}")
print(f"DATA PATH <#################################################3 {data_path}")
print(f"DATA PATH <#################################################3 {data_path}")
print(f"DATA PATH <#################################################3 {data_path}")







## Dataset registration

from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec

assets_directory = pathlib.Path.cwd() / "torch_fedavg_assets"
dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}

for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    client = clients[org_id]

    permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

    # DatasetSpec is the specification of a dataset. It makes sure every field
    # is well-defined, and that our dataset is ready to be registered.
    # The real dataset object is created in the add_dataset method.

    dataset = DatasetSpec(
        name="MNIST",
        data_opener=assets_directory / "dataset" / "mnist_opener.py",
        description=assets_directory / "dataset" / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )
    dataset_keys[org_id] = client.add_dataset(dataset)
    assert dataset_keys[org_id], "Missing dataset key"

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=data_path / f"org_{i+1}" / "train",
    )
    train_datasample_keys[org_id] = client.add_data_sample(data_sample)

    # Add the testing data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_keys[org_id]],
        path=data_path / f"org_{i+1}" / "test",
    )
    test_datasample_keys[org_id] = client.add_data_sample(data_sample)

logger.info("Data registered successfully.")
logger.info(f"Dataset keys: {dataset_keys}")
logger.info(f"Train datasample keys: {train_datasample_keys}")
logger.info(f"Test datasample keys: {test_datasample_keys}")



## Metrics definition
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np


def accuracy(data_from_opener, predictions):
    y_true = data_from_opener["labels"]

    return accuracy_score(y_true, np.argmax(predictions, axis=1))


def roc_auc(data_from_opener, predictions):
    y_true = data_from_opener["labels"]

    n_class = np.max(y_true) + 1
    y_true_one_hot = np.eye(n_class)[y_true]

    return roc_auc_score(y_true_one_hot, predictions)

print(f"Metrics definition. Function accuracy >>>>>>>>>>>>>>>>>>>>>>>> {accuracy}")
print(f"Metrics definition. Function roc_auc >>>>>>>>>>>>>>>>>>>>>>>> {roc_auc}")











## Model definition
import torch
from torch import nn
import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, eval=False):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=not eval)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=not eval)
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=not eval)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


print(f"Model definition. Complete")











## Specifying on how much data to train

from substrafl.index_generator import NpIndexGenerator

# Number of model updates between each FL strategy aggregation.
NUM_UPDATES = 100

# Number of samples per update.
BATCH_SIZE = 32

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)


print(f"Specifying on how much data to train. Complete")





## Torch Dataset definition
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, data_from_opener, is_inference: bool):
        self.x = data_from_opener["images"]
        self.y = data_from_opener["labels"]
        self.is_inference = is_inference

    def __getitem__(self, idx):
        if self.is_inference:
            x = torch.FloatTensor(self.x[idx][None, ...]) / 255
            return x

        else:
            x = torch.FloatTensor(self.x[idx][None, ...]) / 255

            y = torch.tensor(self.y[idx]).type(torch.int64)
            y = F.one_hot(y, 10)
            y = y.type(torch.float32)

            return x, y

    def __len__(self):
        return len(self.x)


print(f"Torch Dataset definition. Complete")















## SubstraFL algo definition

from substrafl.algorithms.pytorch import TorchFedAvgAlgo


class TorchCNN(TorchFedAvgAlgo):
    def __init__(self):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TorchDataset,
            seed=seed,
            use_gpu=False,
        )

print(f"SubstraFL algo definition. Complete")











## Federated Learning strategies
from substrafl.strategies import FedAvg

strategy = FedAvg(algo=TorchCNN(), metric_functions={"Accuracy": accuracy, "ROC AUC": roc_auc})


print(f"Federated Learning strategies. Complete")














## Where to train where to aggregate
from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode


aggregation_node = AggregationNode(ALGO_ORG_ID)

# Create the Train Data Nodes (or training tasks) and save them in a list
train_data_nodes = [
    TrainDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    for org_id in DATA_PROVIDER_ORGS_ID
]

print(f"Where to train where to aggregate. Complete")













## Where and when to test
from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy

# Create the Test Data Nodes (or testing tasks) and save them in a list
test_data_nodes = [
    TestDataNode(
        organization_id=org_id,
        data_manager_key=dataset_keys[org_id],
        data_sample_keys=[test_datasample_keys[org_id]],
    )
    for org_id in DATA_PROVIDER_ORGS_ID
]


# Test at the end of every round
my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, eval_frequency=1)


print(f"Where and when to test. Complete")





# Running the experiment
## specify the third parties dependencies required to run it
from substrafl.dependency import Dependency

dependencies = Dependency(pypi_dependencies=["numpy==1.24.3", "scikit-learn==1.3.1", "torch==2.0.1", "--extra-index-url https://download.pytorch.org/whl/cpu"])

print(f"specify the third parties dependencies required to run it. Complete")




## execute_experiment
from substrafl.experiment import execute_experiment
import logging
import substrafl

substrafl.set_logging_level(loglevel=logging.ERROR)
# A round is defined by a local training step followed by an aggregation operation
NUM_ROUNDS = 10

print(f"execute_experiment. Started ")


compute_plan = execute_experiment(
    client=clients[ALGO_ORG_ID],
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=dependencies,
    clean_models=False,
    name="MNIST documentation example",
)


print(f"execute_experiment. Complete  ")


# The results will be available once the compute plan is completed
client_0.wait_compute_plan(compute_plan.key)






## List results
import pandas as pd

performances_df = pd.DataFrame(client.get_performances(compute_plan.key).model_dump())
print("\nPerformance Table: \n")
print(performances_df[["worker", "round_idx", "identifier", "performance"]])

print(f"List results. Complete  ")







## Plot results
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Test dataset results")

axs[0].set_title("Accuracy")
axs[1].set_title("ROC AUC")

for ax in axs.flat:
    ax.set(xlabel="Rounds", ylabel="Score")


for org_id in DATA_PROVIDER_ORGS_ID:
    org_df = performances_df[performances_df["worker"] == org_id]
    acc_df = org_df[org_df["identifier"] == "Accuracy"]
    axs[0].plot(acc_df["round_idx"], acc_df["performance"], label=org_id)

    auc_df = org_df[org_df["identifier"] == "ROC AUC"]
    axs[1].plot(auc_df["round_idx"], auc_df["performance"], label=org_id)

plt.legend(loc="lower right")
plt.show()

print(f"Plot results. Complete  ")


## Download a model
from substrafl.model_loading import download_algo_state

client_to_download_from = DATA_PROVIDER_ORGS_ID[0]
round_idx = None

algo = download_algo_state(
    client=clients[client_to_download_from],
    compute_plan_key=compute_plan.key,
    round_idx=round_idx,
)

model = algo.model

print(model)
print(f"Download a model. Complete  ")
```

![Run 19 08 2024](https://github.com/user-attachments/assets/79cbe97c-544b-4232-8c91-7586f92235e0)


# Issue retro steps 
The goal of this repo is to share local implementation for the [Using-Torch-FedAvg-on-MNIST-dataset tutorial](https://docs.substra.org/en/stable/examples/substrafl/get_started/run_mnist_torch.html#List-results)

The tutorial was implemented up to the [List results section (including the "List results")](https://docs.substra.org/en/stable/examples/substrafl/get_started/run_mnist_torch.html#List-results)


## Retro steps 
1) CD repo 
```
cd C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset #swap with your repo path 
```

2) Create a new env 

```
conda env create -f substra-environment-torch_fedavg_assets.yml
```
3) Activate the env 
```
conda activate torch_fedavg_assets

```

4) Run the app 
```
python -m src.main
```

5) Get the error: 


```
(torch_fedavg_assets) C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset>python -m src.main
Number of organizations: 3
Algorithm provider: MyOrg1MSP
Data providers: ['MyOrg2MSP', 'MyOrg3MSP']
MNIST data prepared successfully.
Data registered successfully.
Dataset keys: {'MyOrg2MSP': '8225bd51-7fd4-4fae-a3d9-ac5c5f9ef075', 'MyOrg3MSP': 'fa86b75e-e0a6-43d3-b0cf-0f3000517b2b'}
Train datasample keys: {'MyOrg2MSP': '70373be7-6870-4674-978a-871ff315660f', 'MyOrg3MSP': '4e46c822-9b92-45ab-a3e4-a5ecc785cb8a'}
Test datasample keys: {'MyOrg2MSP': 'b2617eca-a93e-401d-a4b7-3f731bfb26a9', 'MyOrg3MSP': 'f9dadf77-e0d2-4cbe-9301-5c531ab817a3'}
Strategy, nodes, and evaluation components created successfully.
Rounds progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 3016.76it/s]
Compute plan progress:   0%|                                                                                                                                                                          | 0/21 [00:00<?, ?it/s]Traceback (most recent call last):
  File "C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset\local-worker\tmptznku85x\function.py", line 13, in <module>
    remote_struct = RemoteStruct.load(src=Path(__file__).parent / 'substrafl_internal')
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substrafl\remote\remote_struct.py", line 94, in load
    instance = cloudpickle.load(f)
               ^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'src'
Compute plan progress:   0%|                                                                                                                                                                          | 0/21 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\compute\spawner\subprocess.py", line 114, in spawn
    subprocess.run(py_command, capture_output=False, check=True, cwd=function_dir, env=envs)
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\subprocess.py", line 571, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['C:\\Users\\probl\\anaconda3\\envs\\torch_fedavg_assets\\python.exe', 'C:\\Users\\probl\\Work\\Substra_env\\Using Torch FedAvg on MNIST dataset\\local-worker\\tmptznku85x\\function.py', '@C:\\Users\\probl\\Work\\Substra_env\\Using Torch FedAvg on MNIST dataset\\local-worker\\tmptznku85x\\tmpgcef6o7l\\arguments.txt']' returned non-zero exit status 1.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset\src\main.py", line 59, in <module>
    main()
  File "C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset\src\main.py", line 45, in main
    compute_plan = run_experiment(
                   ^^^^^^^^^^^^^^^
  File "C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset\src\experiment_config.py", line 32, in run_experiment
    return execute_experiment(
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substrafl\experiment.py", line 498, in execute_experiment
    compute_plan = client.add_compute_plan(
                   ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\client.py", line 48, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\client.py", line 548, in add_compute_plan
    return self._backend.add(spec, spec_options=spec_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\backend.py", line 487, in add
    compute_plan = add_asset(spec, spec_options)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\backend.py", line 406, in _add_compute_plan
    compute_plan = self.__execute_compute_plan(spec, compute_plan, visited, tasks, spec_options)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\backend.py", line 269, in __execute_compute_plan
    self.add(
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\backend.py", line 491, in add
    add_asset(key, spec, spec_options)
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\backend.py", line 437, in _add_task
    self._worker.schedule_task(task)
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\compute\worker.py", line 359, in schedule_task
    self._spawner.spawn(
  File "C:\Users\probl\anaconda3\envs\torch_fedavg_assets\Lib\site-packages\substra\sdk\backends\local\compute\spawner\subprocess.py", line 116, in spawn
    raise ExecutionError(e)
substra.sdk.backends.local.compute.spawner.base.ExecutionError: Command '['C:\\Users\\probl\\anaconda3\\envs\\torch_fedavg_assets\\python.exe', 'C:\\Users\\probl\\Work\\Substra_env\\Using Torch FedAvg on MNIST dataset\\local-worker\\tmptznku85x\\function.py', '@C:\\Users\\probl\\Work\\Substra_env\\Using Torch FedAvg on MNIST dataset\\local-worker\\tmptznku85x\\tmpgcef6o7l\\arguments.txt']' returned non-zero exit status 1.

(torch_fedavg_assets) C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset>
```

# Used local machine specification 
System Information: OS: Windows 11 10.0.22631 Machine: AMD64

CPU Information: AMD Ryzen 7 7840HS w/ Radeon 780M Graphics 8 cores, 16 threads

Memory Information: Total: 31.22GB

Disk Information: disk_C:: Total: 930.43GB

GPU Information: NVIDIA GeForce RTX 4070 Laptop GPU: Total memory: 8188.0MB
