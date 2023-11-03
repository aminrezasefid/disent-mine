import h5py
import numpy as np
from torch.utils.data.dataloader import default_collate
class HDF5:
    def __init__(self,path,factor_names,dataset_name):
        self.path=path
        self.factor_names=factor_names
        self.dataset_name=dataset_name
        hf = h5py.File(path, 'r')
        factor_data_list=[]
        for factor_name in factor_names:
            factor_data_list.append(hf[factor_name][:].reshape(-1,1))
        self.factor_dataset=np.concatenate(factor_data_list,axis=1)
        self.dataset=hf[dataset_name][:]
    def dataset_sample(self, num_samples: int, mode: str, collate: bool = True):
        """Sample a batch of observations X and factors Y."""
        indices=np.random.randint(0,len(self.dataset),num_samples)
        factors = self.factor_dataset[indices]
        batch = self.dataset[indices]
        return default_collate(batch), (default_collate(factors) if collate else factors)
        
