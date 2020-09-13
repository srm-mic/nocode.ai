import torch  
import torch.nn as nn  

from .utils.empty_layer import EmptyLayer

import re   

class NocodeWrapper(object):

    def __init__(self, _id, prev_node, ops, *args):
 
        self.prev = prev_node 
        self.ops = ops 
        self.args = args
        self._id = _id
        self._supported_modules = {
            "conv2d": "Conv2d",
            "Linear": "Linear", 
            "relu": "ReLU", 
            "softmax": "Softmax", 
            "concat": "cat"
        }

        self.node = self._get_torch_module(self.ops, args)
    

    def _get_torch_module(self, module_name, args):

        split_equals = lambda x: x.split("=")
        
        module_name = re.sub("[^a-z1-2]", "", module_name) 

        arguments = {}
        if len(args[0]) > 0:
            
            try:
                args = args[0].split(" ")
            except:            
                args = args[0][0].split(" ")

            for a in args:

                name, value = split_equals(a)
                name = re.sub("[^a-z_]", "", name)
                value = re.sub("[^0-9_\[a-z:0-9\]]", "", value)

                if module_name != "concat":
                    exec(f"{name}={int(value)}")
                else:
                    if name == "dim":       
                        exec(f"{name}={int(value)}")             
                    else:
                        exec(f"{name}={value.strip('[]').split(':')}")
                    

                arguments.update({name: locals()[name]})
        
        try:
            _module = getattr(nn, self._supported_modules[module_name])(**arguments)
        except:
            _module = EmptyLayer()
            print("Empty layer added for args: ", arguments)
            
        return _module
