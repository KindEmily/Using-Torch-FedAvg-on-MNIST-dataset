# src/main.py

from .organizations import clients, ORGS_ID, ALGO_ORG_ID, DATA_PROVIDER_ORGS_ID
from .prepare_data import prepare_mnist_data
from .dataset_registration import register_data

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

if __name__ == "__main__":
    main()