from src.parser import parse_yaml
from src.wrapper import NocodeWrapper

import torch  
import torch.nn as nn 
import torch.nn.functional as F  


class SkeletonModel(nn.Module):

    def __init__(self, forward_list, custom_list={}, _debug=False):
        
        super(SkeletonModel, self).__init__()

        #for key, val in forward_list.items():
            #print(f"{key}-> prev_node: {val.prev}, used_later: {val._used_later}")
        
        self.forward_list = forward_list
        self.custom_list = custom_list
        self._reserved = {}
        self._current_active_node = None
        self._debug = _debug

    def load_weights(self, path):

        incoming_state_dict = torch.load(path)
        #print(incoming_state_dict.keys())

        for key, val in self.forward_list.items():
            _tmp_state_dict = {}
            for k, v in val.node.named_parameters():
                _tmp_state_dict.update({k:incoming_state_dict[key+'.'+k]})
            
            val.node.load_state_dict(_tmp_state_dict)
            self.forward_list.update({key:val})

    def load_state_dict(self):
        raise NotImplementedError("please use load_weights() fn")

    def show_nodes(self):
        for k, v in self.forward_list.items():
            print(f"{k}: {v.node}")

    def forward(self, x):

        for key, node in self.forward_list.items():

            if self._debug:
                print(f"key:{key}, prev_node: {node.prev}, used_later:{node._used_later}, ops:{node.ops}")

            if self._current_active_node is not None: 
                
                if self._current_active_node != node.prev:

                    x = self._reserved[node.prev]

            if node.ops in ["conv2d" , "relu", "softmax", "linear", "sigmoid"]:
                
                #print(f"Current_id: {node._id}, prev_node: {node.prev}")
                x = node.node(x)
                self._current_active_node = node._id
                
                if node._used_later == True:
                    #print(key)        
                    self._reserved.update({key: x})
            
            elif node.ops == "concat":
                #print(f"Current_id: {node._id}, prev_node: {node.prev}")
                _to_be_concat = [self._reserved[i] for i in node.args["tensors"]]
                x = torch.cat(_to_be_concat, dim=node.args["dim"])
                self._current_active_node = node._id
            
            elif node.ops == "custom":

                _tmp_node = self.custom_list[node.args["block"]]
                #print(_tmp_node)
                x = _tmp_node(x)
                self._current_active_node = node._id

                if node._used_later == True:
                    self._reserved.update({key, x})
          
                del _tmp_node

        return x
                

        



        















def build_model(forward_pass):
    raise NotImplementedError


