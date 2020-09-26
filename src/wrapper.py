import torch  
import torch.nn as nn  

from .utils.empty_layer import EmptyLayer

import re   

class NocodeWrapper(object):

    def __init__(self, _id, prev_node, ops, *args):
 
        self.prev = re.sub("[^a-z0-9]", "", prev_node) 
        self.ops = re.sub("[^a-z0-9]", "", ops) 
        self.args = args
        self._id = re.sub("[^a-z0-9]", "", _id) 
        self._supported_modules = {
            "conv2d": "Conv2d",
            "Linear": "Linear", 
            "relu": "ReLU", 
            "softmax": "Softmax", 
            "concat": "cat", 
            "linear": "Linear", 
            "sigmoid": "Sigmoid", 
            "flatten": "flatten", 
            "custom": "custom"

        }
        self._used_later = False
        self._require_previous = False
        self.node = self._get_torch_module(self.ops, args)
    

    def _get_torch_module(self, module_name, args):

        split_equals = lambda x: x.split("=")
        module_name = re.sub("[^a-z0-9]", "", module_name) 

        if module_name != "flatten":
            _parent_torch_module = nn 
        else:
            _parent_torch_module = torch

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

                if module_name != "concat" and module_name != "custom":
                    exec(f"{name}={int(value)}")
                
                elif module_name == "custom":
                    exec(f"{name}='{value}'")
                
                else:
                    
                    self._require_previous = True

                    if name == "dim":       
                        exec(f"{name}={int(value)}")             
                    else:
                        exec(f"{name}={value.strip('[]').split(':')}")
                    

                arguments.update({name: locals()[name]})
        
        try:
            _module = getattr(_parent_torch_module, self._supported_modules[module_name])(**arguments)
        except:
            _module = EmptyLayer()
            self.args = arguments
            print(f"Empty layer added for {module_name} args: {arguments}")
        
        return _module