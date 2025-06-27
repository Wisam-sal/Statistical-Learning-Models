import json
from typing import Dict, Any, List
from pathlib import Path
import time
from functools import wraps
import pyperclip
import os
from bs4 import BeautifulSoup

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file from the given path.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def list_files_in_folder(folder_path: str) -> List[str]:
    """
    List all file paths in the specified folder using Path.
    """
    return [str(file) for file in Path(folder_path).iterdir()]# if file.is_file()]

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Calling {func.__name__}...")
        result = func(*args, **kwargs)  
        end_time = time.time()  
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper
