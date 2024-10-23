import os
import yaml

def get_configs() -> dict:
    """Returns configs dictionary"""

    config_file_path = os.path.join(os.path.dirname(__file__), "..", "config/config.yaml")

    with open(os.path.abspath(config_file_path), 'r') as file:
        config = yaml.safe_load(file)

    return config