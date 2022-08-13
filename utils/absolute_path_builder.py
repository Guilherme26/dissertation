import yaml
import os

class AbsolutePathBuilder():
    @staticmethod
    def get_path(dataset, filepaths="../config/filepaths.yaml"):
        with open(filepaths) as f:
            filepaths = yaml.safe_load(f)
        
        if dataset not in filepaths:
            raise ValueError(f"Dataset `{dataset}` doesn't exist.")
        
        return os.path.join(filepaths["base_path"], filepaths[dataset])
