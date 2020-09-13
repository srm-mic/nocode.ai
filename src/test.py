from wrapper import NocodeWrapper

class Item(NocodeWrapper):
    
    def __init__(self):
        super(Item, self).__init__(1, 3, 4, ops="conv2d", a=5)
        print(self.args)
        print(self._get_torch_module("Linear", in_features=2, out_features=2))
i = Item()