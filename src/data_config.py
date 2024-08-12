# src/data_config.py

from substrafl.index_generator import NpIndexGenerator

# Number of model updates between each FL strategy aggregation.
NUM_UPDATES = 100

# Number of samples per update.
BATCH_SIZE = 32

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)

# Import these from your existing files
from .organizations import ALGO_ORG_ID, DATA_PROVIDER_ORGS_ID
from .dataset_registration import register_data