import yaml
import os

# Read yaml file — resolve path relative to this file, not CWD
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    config_path = os.path.join(_MODEL_DIR, 'config.yaml')
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config
