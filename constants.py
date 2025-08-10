# All folder 
from pathlib import Path

#Data directory and files
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "iris.csv"

#model directory and file
MODEL_DIR = Path("model")
MODEL_FILE = MODEL_DIR / "iris_model.joblib"

#Training configuration
TEST_SIZE = 0.33
RANDOM_STATE = 21


