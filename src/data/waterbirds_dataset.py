from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import h5py
import random
import os 
class WaterbirdsDataset(Dataset):
    def __init__(self,root,split, transform=None):
        file = os.path.join(root, "Waterbirds", "waterbirds_dataset.h5py")
        if split == "all":
            split = "test"
            
        self.file_object=h5py.File(file,'r')
        self.split = split
        self.dataset=self.file_object['Waterbirds'][split]
        self.num_data = len(self.dataset)
        self.index_map = list(range(self.num_data))
        self.transform = transform
        pass

    def __len__(self):
        return self.num_data

    def __getitem__(self,index):
        if(index >= self.num_data):
            raise IndexError()
        true_idx = self.index_map[index]
        img = self.dataset[str(true_idx)][()]
        if self.transform is not None:
            img = self.transform(img)
        y = self.dataset[str(true_idx)].attrs['y']
        return img, y, self.dataset[str(true_idx)].attrs["place"]
    
    def set_dataset_size(self, subset_size):
        indices = list(range(self.num_data))
        random.shuffle(indices)
        self.index_map = [self.index_map[i] for i in indices[:subset_size]]
        self.num_data = subset_size
        return self.num_data
        
    def switch_mode(self, original, rotation):
        pass

    def plot_image(self,index):
        true_idx = self.index_map[index]
        img = self.dataset[str(true_idx)][()]
        if self.transform is not None:
            img = self.transform(img)
            
        plt.imshow(img.permute(1,2,0).numpy(),interpolation='nearest')
        pass
    pass