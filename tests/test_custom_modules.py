from src.build_model import build_model, SkeletonModel
from src.parser import parse_yaml
import torch

network, custom = parse_yaml("demo/demo_custom_modules.yaml")

#for k, v in network.items():
#    print(f"{k}: {v.node}")
for k, v in custom.items():
    print(f"==={k}===")
    for key, val in v.items():
       print(f"{key}: {val.node}")
    v = SkeletonModel(v, _debug=True)
    custom.update({k:v})
x = torch.ones((1, 3, 10, 10))
model = SkeletonModel(network, custom, _debug=True)
out = model(x)
model.show_nodes()
pretrained_model = torch.load("demo/DemoCustomModule.pth")
for k, v in pretrained_model.items():
    print(k)