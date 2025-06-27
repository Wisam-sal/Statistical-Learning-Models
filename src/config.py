import sys
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Data Roots
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "rawdata"
DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data Endings
SPACE_MAP_ROOT = 'map.png'
GROUND_TRUTH_ROOT = 'ground_truth.json'
CAMERA_CALIBRATION_ROOT = 'calibration.json'
