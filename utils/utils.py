import os
import yaml

class getUtils:

    def load_yaml(self):
        path = "config/config.yaml"
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML not found at: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)