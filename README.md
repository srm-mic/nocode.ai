# nocode.ai

Design, Build, Train and Deploy AI without any code!

[ Under Active Development ]

## Immediate todos

- [x] Verify forward pass

- [ ] Only last node in forward pass returns. As in multi return forward pass
      note supported.

- [x] Model loading
    
- [ ] Unbranched linear custom modules support
      
- [ ] DO NOT take the *look up in custom module dict approach, in that case copies are used*. Instead figure out a way have individual entities for each custom module occurance in the forward pass itself.
      
- [ ] Custom modules verification
      
- [ ] Weights loading
      
- [ ] Branching in Custom Modules

- [ ] More Ops

- [ ] CLI

- [ ] Training API and/or from CLI

## Support to be added

- [ ] Unnamed parameters in yaml file

- [ ] Negative indices in concat/add operations

- [ ] Maybe remove node attribute from wrapper after adding to module list

- [ ] Remove the necessity of the first entry in YAML File to be identifier for ops like concat

- [x] **Memory efficiency: save intermediate steps only when required by a later node**
