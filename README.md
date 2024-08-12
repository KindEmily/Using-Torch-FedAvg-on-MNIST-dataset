Create the environment from the `substra-environment.yml` file:

<aside>
💡 Protip: dont forget to set the `name:` property with something meaningful
</aside>

```
cd C:\Users\probl\Work\Substra_env\Using Torch FedAvg on MNIST dataset
```

```
conda env create -f substra-environment-torch_fedavg_assets.yml
```


# Important - no GPU in current version 
```
The CPU torch version is installed here to have a Dependency object as light as possible as we don’t use GPUs (use_gpu set to False). Remove the --extra-index-url to install the cuda torch version.
```
Source: https://docs.substra.org/en/stable/examples/substrafl/get_started/run_mnist_torch.html#Running-the-experiment

# Optionally
## Optionally, you might want to add some configuration options for the experiment in src/data_config.py:

```
# src/data_config.py

# ... (previous content remains the same)

# Evaluation configuration
EVAL_FREQUENCY = 1  # Test at the end of every round
```

## Then, you can use this configuration in evaluation_config.py:
```
# src/evaluation_config.py

from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy
from .data_config import DATA_PROVIDER_ORGS_ID, EVAL_FREQUENCY

# ... (rest of the file remains the same)

def create_evaluation_strategy(test_data_nodes, eval_frequency=EVAL_FREQUENCY):
    return EvaluationStrategy(
        test_data_nodes=test_data_nodes, 
        eval_frequency=eval_frequency
    )
```

## Optionally, you might want to add some configuration options for the experiment in src/data_config.py:



```
# src/data_config.py

# ... (previous content remains the same)

# Experiment configuration
NUM_ROUNDS = 3
EXPERIMENT_NAME = "MNIST documentation example"
```

## Then, you can use this configuration in experiment_config.py:
```
# src/experiment_config.py

from .data_config import NUM_ROUNDS, EXPERIMENT_NAME

# ... (rest of the file remains the same)

def run_experiment(
    client,
    strategy,
    train_data_nodes,
    evaluation_strategy,
    aggregation_node
):
    return execute_experiment(
        client=client,
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=evaluation_strategy,
        aggregation_node=aggregation_node,
        num_rounds=NUM_ROUNDS,
        experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
        dependencies=dependencies,
        clean_models=False,
        name=EXPERIMENT_NAME,
    )
```