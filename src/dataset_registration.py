# src/data_registration.py

import pathlib
from substra.sdk.schemas import DatasetSpec, Permissions, DataSampleSpec
from .organizations import DATA_PROVIDER_ORGS_ID, ALGO_ORG_ID, clients

def register_data(data_path):
    assets_directory = pathlib.Path(__file__).parent.parent / "torch_fedavg_assets"
    dataset_keys = {}
    train_datasample_keys = {}
    test_datasample_keys = {}

    for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
        client = clients[org_id]

        permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

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

    return dataset_keys, train_datasample_keys, test_datasample_keys