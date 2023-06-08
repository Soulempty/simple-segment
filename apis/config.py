import os
import yaml
import codecs
from typing import Dict


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True
    
class Config(object):

    def __init__(self,path):
        assert os.path.exists(path), 'Config path ({}) does not exist'.format(path)
        self.dic = self._parse_from_yaml(path)
        for k,v in self.dic.items():
            setattr(self,k,v)

    @property       
    def data(self):
        return self.dic
    
    @classmethod
    def _parse_from_yaml(self,path):
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
            return dic

    def __str__(self) -> str:
        return yaml.dump(self.dic, Dumper=NoAliasDumper)



