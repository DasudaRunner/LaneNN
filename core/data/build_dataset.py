from .dataset import SynfaceDataset
import pdb

def build_data(C):
    config = C.config
    ginfo = C.ginfo
    
    tasks = config['tasks']
    this_task = tasks[ginfo.task_id]
    
    dataset_kwargs = config['dataset']
    dataset_kwargs.update(this_task)
    
    dataset = SynfaceDataset(dataset_kwargs)
    
    # if link.get_rank()==0:
    #     for i in range(10):
    #         out = dataset[i]
    # pdb.set_trace()
    
    return dataset