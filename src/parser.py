import yaml

import torch
import torch.nn as nn

from .wrapper import NocodeWrapper
import re

""" 
Parse the YAML files in a ready to build format. 
Handles all the parsing issues i.e. taking care 
of all skip / concat connections and other similar issues.

The output of this file is such that it's read-to-use. You 
literally have to iterate through the items and build the neural network
"""

def parse_yaml(path):
    """ 
    Parses the YAML-file and returns the model configs 
    in an easy to build format.

    Details: 

    The function parses the YAML File and constructs a graph 
    modelled as a dict representation internally and returns it 
    for construction. For example, for a given config file it can 
    return:
    {
        "A": ["B"], 
        "B": ["C", "D"], 
        "C": ["D"], 
        "D" : []
    }

    Arguments: 
        -path: path to the YAML config file 
    
    Returns: 
        -final_config: A dict containing the final objects 
    """ 

    # load the yaml file
    with open(path, "r") as fs: 

        try:        
            configfile = yaml.safe_load(fs)
        except yaml.YAMLError as err:
            print(err)
            exit()
    
    model = nn.ModuleList()
    forward_pass = {}

    network = configfile["network"]
    model_name = configfile["name"]

    def _build_module_list(network):

        _prev_node = None
    
        for items in network:

            if isinstance(items, dict):
                
                _build_module_list([items["branch"][0]])
                continue

            _items_split = items.split(" ", 2)
            
            _id = _items_split[0]

            _ops = _items_split[1]
            
            try:
                _args = _items_split[2:]
            except:
                pass

            _prev_node = _id

            # flow control block
            _node_wrapper = NocodeWrapper(_id, _prev_node, _ops, _args)

            #_id = re.sub("[^a-z1-2]", "", _id) # remove this. repeted in Nocodewrapper

            # appending current wrapper object to forward pass list
            forward_pass.update({_node_wrapper._id : _node_wrapper})

            # changing the parity bits of the all the nodes that are required 
            # later by this node
            if _node_wrapper._require_previous == False:
                pass 
            else:
                # right now only supports concat. Basically sets all the _required_later
                # parity of all the tensors involved as True so that they can be selectively
                # stored to save memory  
                for node in _node_wrapper.args['tensors']:
                    forward_pass[node]._used_later = True


        
    _build_module_list(network)

    return forward_pass
    

    


