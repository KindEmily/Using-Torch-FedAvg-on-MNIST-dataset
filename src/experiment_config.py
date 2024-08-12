# src/experiment_config.py

import pathlib
import logging
from substrafl.experiment import execute_experiment
from substrafl.dependency import Dependency
import substrafl

# Set logging level
substrafl.set_logging_level(loglevel=logging.ERROR)

# Define dependencies
dependencies = Dependency(
    pypi_dependencies=[
        "numpy==1.24.3",
        "scikit-learn==1.3.1",
        "torch==2.0.1",
        "--extra-index-url https://download.pytorch.org/whl/cpu"
    ]
)

# Number of rounds
NUM_ROUNDS = 3

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
        name="MNIST documentation example",
    )