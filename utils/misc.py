import json
import yaml
import os
import pickle as pkl

class IncludeLoader(yaml.Loader):
    def __init__(self, *args, **kwargs):
        super(IncludeLoader, self).__init__(*args, **kwargs)
        self.add_constructor('!include', self._include)
        self.root = os.path.curdir

    def _include(self, loader, node):
        oldRoot = self.root
        filename = os.path.join(self.root, loader.construct_scalar(node))
        self.root = os.path.dirname(filename)
        data = yaml.load(open(filename, 'r'))
        self.root = oldRoot
        return data

def load_yaml(file_path):
    with open(file_path) as f:
        file_con = yaml.load(f, Loader=IncludeLoader)
    return file_con

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
def save_json(dat, file_name):
    with open(file_name, "w") as file:
        json.dump(dat, file, indent=4)

def read_list(file_path):
    with open(file_path, 'r') as f:
        dat = f.readlines()
    dat = [i.replace('\n','') for i in dat]
    return dat
def save_list(dat, file_name):
    new_dat = []
    for i in dat:
        if i.endswith('\n'):
            new_dat.append(str(i))
        else:
            new_dat.append(str(i)+'\n')
    with open(file_name,'w') as f:
        f.writelines(new_dat)

def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        data = pkl.load(f)
    return data