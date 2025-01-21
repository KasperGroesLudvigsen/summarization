import yaml

# Load baseline configuration
with open("training_configs/config1.yml", "r") as file:
    config = yaml.safe_load(file)

print(config)

wandb_config = config["wandb_config"]

import wandb

wandb.init(project="config_example2",
           config=config)