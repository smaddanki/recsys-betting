from pathlib import Path
import torch
from pydantic_settings import BaseSettings

if torch.cuda.is_available():
    device = "cuda" 
elif torch.backends.mps.is_available():
    device="mps"
else:
    device = "cpu"


class Settings(BaseSettings):
    
    DEVICE: str = device
    
    RECSYS_DIR: Path = Path(__file__).parent
    
    DATA_DIR: Path = Path(__file__).parents[1] / 'data'
    
    MODEL_REGISTRY: Path = Path(__file__).parents[1] / 'model_registry'
    
    SOURCE_DATA_DIR: Path = DATA_DIR / 'raw'
    
    PROCESSED_DATA_DIR: Path = DATA_DIR / 'processed'

    # Training
    TWO_TOWER_MODEL_EMBEDDING_SIZE: int = 16
    TWO_TOWER_MODEL_BATCH_SIZE: int = 2048
    TWO_TOWER_NUM_EPOCHS: int = 10
    TWO_TOWER_WEIGHT_DECAY: float = 0.001
    TWO_TOWER_LEARNING_RATE: float = 0.01
    TWO_TOWER_DATASET_VALIDATON_SPLIT_SIZE: float = 0.1
    TWO_TOWER_DATASET_TEST_SPLIT_SIZE: float = 0.1

settings = Settings()