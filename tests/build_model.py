from src.build_model import build_model, SkeletonModel
from src.parser import parse_yaml
from PIL import Image
import torch   
from torchvision import transforms
import torch.nn as nn

network = parse_yaml("demo/demo_config.yaml")

sm  = SkeletonModel(network)
sm.load_weights("test.pth")
#for key, value in sm.named_modules():
    #print(key)
#load_from = torch.load("test.pth")


#for key, val in load_from.items():
    #pass
ex = torch.ones((1, 3, 10, 10))
ans = sm(ex)
print(ans[0, 0])
print(ans[0, 1])
print(ans[0, 2])