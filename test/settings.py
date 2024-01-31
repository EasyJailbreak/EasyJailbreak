import os
from pathlib import Path

current_path = Path(__file__).resolve().parent
print(current_path)

CSV_DATA_PATH = os.path.join(current_path, 'mini.csv')
JSON_DATA_PATH = os.path.join(current_path, 'mini.jsonl')
