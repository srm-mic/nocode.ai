import torch.nn as nn  

params = {
    "in_features": 2, 
    "out_features": 2
}
l1 = nn.Linear(**params)

class demo(nn.Module):
    def __init__(self):
        super(demo, self).__init__()
        self.layer = nn.Linear(2, 2)
    def forward(self, x):
        return self.layer(x)

l2 = demo()

print(l1)
print(type(l2))
