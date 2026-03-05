import os
from typing import Callable
from config import prefix, root_dir
import regex as re

def _get_pipeline(getter: Callable):
    existing_dirs = os.listdir(root_dir)
    indexes = [re.search(f"{prefix}_(\\d+)", dir) for dir in existing_dirs]
    indexes = [int(x.group(1)) for x in indexes if x]
    indexes.append(0)
    try:
        index = getter(indexes)
    except Exception as e:
        print(f"Could not find pipeline: {e}")
    pipeline_name = f"{prefix}_{index}"
    os.path.join(root_dir, pipeline_name)
    
def pipeline_path(num: int):
    return _get_pipeline(lambda x: x.index(num))

def recent_pipeline_path():
    return _get_pipeline(max)

def next_pipeline_path():
    return _get_pipeline(lambda x: max(x) + 1)