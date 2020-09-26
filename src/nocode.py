from .build_model import SkeletonModel
from .parser import parse_yaml


def Build(path):
    network = parse_yaml(path)

    """
    # wrap custom modules around with SkeletonModel
    for k, v in custom.items():
        for key, val in v.items():
            #print(f"{key}: {val.node}")
            v = SkeletonModel(v)
        custom.update({k:v})
    """
    
    model = SkeletonModel(network)
    
    return model