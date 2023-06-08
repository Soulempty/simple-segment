import inspect
from collections.abc import Sequence
import warnings

class Registry:
    def __init__(self, name=None):
        self._module_dict = dict()
        self._name = name

    def __len__(self):
        return len(self._module_dict)

    def __repr__(self):
        name_str = self._name if self._name else self.__class__.__name__
        return "{}:{}".format(name_str, list(self._module_dict.keys()))

    def __getitem__(self, item):
        if item not in self._module_dict.keys():
            raise KeyError("{} does not exist in availabel {}".format(item,self))
        return self._module_dict[item]

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def name(self):
        return self._name

    def _register_module(self, module):
       
        # Currently only support class or function type
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError("Expect class/function type, but received {}".
                            format(type(module)))

        module_name = module.__name__

        if module_name in self._module_dict.keys():
            warnings.warn("{} exists already! It is now updated to {} !!!".
                          format(module_name, module))
        self._module_dict[module_name] = module

    def register_module(self, module):
        self._register_module(module)

        return module


MODELS = Registry("model")
BACKBONES = Registry("backbone")
DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")
LOSSES = Registry("losses")

 
