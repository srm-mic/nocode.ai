import yaml
from src.parser import parse_yaml

parsed_network = parse_yaml("demo/demo_config.yaml")

for item in parsed_network:
    print(item.node)
