# src/main.py

from .data_config import DATA_PROVIDER_ORGS_ID, ALGO_ORG_ID
from .prepare_data import prepare_mnist_data
from .dataset_registration import register_data
from .strategy_config import create_strategy
from .aggregation_config import create_aggregation_node, create_train_data_nodes
from .evaluation_config import create_test_data_nodes, create_evaluation_strategy
from .experiment_config import run_experiment
from .results_analysis import analyze_results
from .organizations import clients
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

    # Create test data nodes and evaluation strategy
    test_data_nodes = create_test_data_nodes(dataset_keys, test_datasample_keys)
    evaluation_strategy = create_evaluation_strategy(test_data_nodes)

    print("Strategy, nodes, and evaluation components created successfully.")
    
    # Run the experiment
    compute_plan = run_experiment(
        client=clients[ALGO_ORG_ID],
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=evaluation_strategy,
        aggregation_node=aggregation_node
    )

    print(f"Experiment executed successfully. Compute plan key: {compute_plan.key}")

    # Analyze results
    # analyze_results(clients[ALGO_ORG_ID], compute_plan.key)

if __name__ == "__main__":
    main()