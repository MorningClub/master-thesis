import tqdm
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from .ego_dataset import EgoDataset
from .main_dataset import LabeledMainDataset
from .main_dataset import ReformattedMainDataset

def data_loader(data_type, config):

    if data_type == 'ego':
        dataset = EgoDataset(config.data_dir, T=config.ego_traj_len)
    elif data_type == 'main':
        dataset = LabeledMainDataset(config.data_dir, config.config_path)
    elif data_type == 'custom_main':
        dataset = ReformattedMainDataset(config.data_dir, config.config_path)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.seed(42069)
        np.random.shuffle(indices)
        train_indices = indices[:1000000]
        val_indices = indices[1000000:1015000]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, sampler=train_sampler), \
                DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, sampler=val_sampler) 
    else:
        raise NotImplementedError(f'Unknown data type {data_type}')

    return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True) 

__all__ = ['data_loader']
