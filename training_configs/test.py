import yaml

# Load baseline configuration
with open("training_configs/baseline.yml", "r") as file:
    baseline_config = yaml.safe_load(file)

print(baseline_config)