# src/evaluation_config.py

from substrafl.nodes import TestDataNode
from substrafl.evaluation_strategy import EvaluationStrategy
from .data_config import DATA_PROVIDER_ORGS_ID

def create_test_data_nodes(dataset_keys, test_datasample_keys):
    return [
        TestDataNode(
            organization_id=org_id,
            data_manager_key=dataset_keys[org_id],
            data_sample_keys=[test_datasample_keys[org_id]],
        )
        for org_id in DATA_PROVIDER_ORGS_ID
    ]

def create_evaluation_strategy(test_data_nodes, eval_frequency=1):
    return EvaluationStrategy(
        test_data_nodes=test_data_nodes, 
        eval_frequency=eval_frequency
    )