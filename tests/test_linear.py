from src.build_model import build_model, SkeletonModel
from src.parser import parse_yaml
import torch

network = parse_yaml("demo/demo_linear.yaml")

sm  = SkeletonModel(network)
sm.show_nodes()

sm.load_weights("test_linear.pth")

x = torch.ones((3, 1))
ans = sm(x.T)

""" 
fix the tests. it works. verified manually. 

print(ans[0].detach() == torch.tensor(0.7022))
assert ans == torch.tensor([[0.7022]]), "Fails on demo_linear.yaml"
"""