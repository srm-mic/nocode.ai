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
    modules = {}
    custom_modules = {}
    _custom_modules_present = False

    network = configfile["network"]

    model_name = configfile["name"]

    if "custom" in configfile.keys():

        raise NotImplementedError("Custom modules are not supported yet but is a high priority")

        custom_module_network = configfile["custom"]
        _custom_modules_present = True
        #print(custom_module_network)
        for item in custom_module_network:

            for k, v in item.items():
                _tmp = _build_custom_module({}, v)
                #print(k)
                custom_modules.update({k: _tmp})

    

    
    def _build_module_list(network, branched_from=None):
        
        if branched_from is None:
            _prev_node = None
        else:
            _prev_node = branched_from

        for items in network:

            if isinstance(items, dict):
                
                # fix this. Just to make it work. making a copy of _prev_node
                # and removing unwanted characters from it. _prev_node is cleaned
                # within the wrapper. remove this unnecessary dual computation
                _prev_node_copy = re.sub("[^a-z0-9]", "", _prev_node) 

                forward_pass[_prev_node_copy]._used_later = True
                _build_module_list(items["branch"], _prev_node_copy)
                
                continue

            _items_split = items.split(" ", 2)
            
            _id = _items_split[0]

            
            try:
                _ops = _items_split[1]
            except Exception as e:
                print(e)
                print(f"Error occured for {_id}")
                exit()
            
            try:
                _args = _items_split[2:]
            except:
                pass
            
            if _prev_node is None:
                _prev_node = _id

            # flow control block
            _node_wrapper = NocodeWrapper(_id, _prev_node, _ops, _args)
            
            _prev_node = _id
        

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
    

def _build_custom_module(custom_modules_dict, network, branched_from=None):

    if branched_from is None:
        _prev_node = None
    else:
        _prev_node = branched_from
    #print("items2:", network)
    for items in network:
        
        if isinstance(items, dict):
                
            # fix this. Just to make it work. making a copy of _prev_node
            # and removing unwanted characters from it. _prev_node is cleaned
            # within the wrapper. remove this unnecessary dual computation
            _prev_node_copy = re.sub("[^a-z1-9]", "", _prev_node) 
            #print("items:", items["branch"])
            custom_modules_dict[_prev_node_copy]._used_later = True
            custom_modules_dict =  _build_custom_module(custom_modules_dict, items["branch"], _prev_node_copy)
                
            continue
        
        _items_split = items.split(" ", 2)

        _id = _items_split[0]

        try:
            _ops = _items_split[1]
        except Exception as e:
            print(e)
            print(f"Error occured for {_id}")
            exit()
            
        try:
            _args = _items_split[2:]
        except:
            pass
            
        if _prev_node is None:
            _prev_node = _id

        # flow control block
        _node_wrapper = NocodeWrapper(_id, _prev_node, _ops, _args)
            
        _prev_node = _id
        

        # appending current wrapper object to forward pass list
        custom_modules_dict.update({_node_wrapper._id : _node_wrapper})

        # changing the parity bits of the all the nodes that are required 
        # later by this node
        if _node_wrapper._require_previous == False:
            pass 
        else:
            # right now only supports concat. Basically sets all the _required_later
            # parity of all the tensors involved as True so that they can be selectively
            # stored to save memory  
            for node in _node_wrapper.args['tensors']:
                custom_modules_dict[node]._used_later = True


    return custom_modules_dict



