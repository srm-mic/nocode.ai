import yaml
from src.parser import parse_yaml

parsed_network = parse_yaml("demo/demo_config.yaml")

for key, item in parsed_network.items():
    print(f"{key}, {item.node}, {item._require_previous}, {item._used_later}, {item.prev}")
