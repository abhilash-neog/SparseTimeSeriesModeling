import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self, file):
        super(MySet, self).__init__()
        json_path = "/raid/abhilash/BRITS/json/"
        with open(json_path+file, 'r') as f:
            self.content = f.readlines()
        indices = np.arange(len(self.content))
        val_indices = np.random.choice(indices, len(self.content) // 5)
        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        rec['is_train'] = 0 if idx in self.val_indices else 1
        return rec

def collate_fn(recs):
    forward = [x['forward'] for x in recs]
    backward = [x['backward'] for x in recs]

    def to_tensor_dict(recs):
        values = torch.FloatTensor([r['values'] for r in recs])
        masks = torch.FloatTensor([r['masks'] for r in recs])
        deltas = torch.FloatTensor([r['deltas'] for r in recs])
        evals = torch.FloatTensor([r['evals'] for r in recs])
        eval_masks = torch.FloatTensor([r['eval_masks'] for r in recs])
        forwards = torch.FloatTensor([r['forwards'] for r in recs])
        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}
    ret_dict['is_train'] = torch.FloatTensor([x['is_train'] for x in recs])

    return ret_dict

def get_loader(batch_size=64, shuffle=True, file=None):
    data_set = MySet(file)
    data_iter = DataLoader(dataset=data_set, batch_size=batch_size, num_workers=4, shuffle=shuffle, pin_memory=True, collate_fn=collate_fn)
    return data_iter
