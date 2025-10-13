import json
import yaml

def read_json(file_path):
    """Reads a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def read_yaml(file_path):
    """Reads a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
