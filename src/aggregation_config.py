# src/aggregation_config.py

from substrafl.nodes import TrainDataNode
from substrafl.nodes import AggregationNode


from .data_config import ALGO_ORG_ID, DATA_PROVIDER_ORGS_ID

def create_aggregation_node():
    return AggregationNode(ALGO_ORG_ID)

def create_train_data_nodes(dataset_keys, train_datasample_keys):
    return [
        TrainDataNode(
            organization_id=org_id,
            data_manager_key=dataset_keys[org_id],
            data_sample_keys=[train_datasample_keys[org_id]],
        )
        for org_id in DATA_PROVIDER_ORGS_ID
    ]