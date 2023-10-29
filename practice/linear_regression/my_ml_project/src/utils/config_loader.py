import json
import os

# Load the configuration file
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_path = os.path.join(project_dir, "config.json")
with open(config_path, "r") as file:
    config = json.load(file)

ENVIRONMENT = config.get("ENVIRONMENT", "development")
# ... Load other necessary configurations as needed