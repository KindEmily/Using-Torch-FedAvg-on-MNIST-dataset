# src/main.py

from .data_config import DATA_PROVIDER_ORGS_ID, ALGO_ORG_ID
from .prepare_data import prepare_mnist_data
from .dataset_registration import register_data
from .strategy_config import create_strategy
from .aggregation_config import create_aggregation_node, create_train_data_nodes
from .organizations import ORGS_ID

def main():
    print(f"Number of organizations: {len(ORGS_ID)}")
    print(f"Algorithm provider: {ALGO_ORG_ID}")
    print(f"Data providers: {DATA_PROVIDER_ORGS_ID}")
    
    # Prepare the MNIST data
    data_path = prepare_mnist_data()
    print("MNIST data prepared successfully.")

    # Register the data
    dataset_keys, train_datasample_keys, test_datasample_keys = register_data(data_path)
    print("Data registered successfully.")
    print(f"Dataset keys: {dataset_keys}")
    print(f"Train datasample keys: {train_datasample_keys}")
    print(f"Test datasample keys: {test_datasample_keys}")

    # Create strategy
    strategy = create_strategy(dataset_keys, train_datasample_keys)

    # Create nodes
    aggregation_node = create_aggregation_node()
    train_data_nodes = create_train_data_nodes(dataset_keys, train_datasample_keys)

    # Here you would typically run your experiment using the created components
    # For example:
    # experiment = execute_experiment(strategy, train_data_nodes, aggregation_node)

if __name__ == "__main__":
    main()