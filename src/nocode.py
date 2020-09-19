from .build_model import SkeletonModel
from .parser import parse_yaml


def Build(path):
    network = parse_yaml(path)
    model = SkeletonModel(network)
    return model