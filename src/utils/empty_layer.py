import torch  
import torch.nn as nn  
import torch.nn.functional as F

class EmptyLayer(nn.Module):

    def __init__(self):

        super(EmptyLayer, self).__init__()
        