# Serializer
"""
Utility Module

Responsible for
    - YAML dump/load
    - JSON dump/load
    - numpy save/load

Forbidden responsibilities
    - knowledge of Pipeline, RunArchive, likelihoods
    - computation or plotting
"""


from pathlib import Path
import yaml, json
import numpy as np


def _to_builtin(obj):
    """
    Recursively convert numpy objects into plain Python types
    so YAML/JSON serializers can always represent them.
    """
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def YAML_dump(obj, path):
    """
    Write dictionary to YAML file
    """
    path = Path(path)
    with open(path, "w") as f:
        yaml.safe_dump(_to_builtin(obj), f)

def JSON_dump(obj, path, indent=4):
    """
    Write dictionary to JSON file
    """
    path = Path(path)
    with open(path, "w") as f:
        json.dump(_to_builtin(obj), f, indent=indent)

def array_dump(arr, path):
    """
    Save numpy array to disk
    """
    path = Path(path)
    np.save(path, arr)


def YAML_load(path):
    """
    Loads the YAML file from a given path
    """
    path = Path(path)
    with open(path, "r") as f:
        yaml_file = yaml.safe_load(f)

    return yaml_file
    