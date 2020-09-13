from src.wrapper import NocodeWrapper
import torch
import torch.nn as nn 

# Test: Nocode correctly initializes a singular torch 
#       module. Gonna use a conv2d layer of params: 
#       in_channels=3, out_channels=64, kernel_size=3, 
#       stride=2

node = NocodeWrapper("conv1",  "testNode2", "conv2d", 
                    "in_channels=3, out_channels=64, kernel_size=3")

testconv = nn.Conv2d(3, 64, 3)

assert type(node.node) is type(testconv), "demo conv node and testconv of not same type"