import torch,torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.data[index])
            y = self.transform(self.labels[index])
        else:
            x = self.data[index]
            y = self.labels[index]
        return (x,y)

    def __len__(self):
        return len(self.data)

# Function to divide the dataset into labeled and unlabeled subsets
def divide_dataset(dataset, labeled_len, seed=0, transform=None):
    # Use the same seed for reproducibility
    torch.manual_seed(seed)
    
    indexer = torch.randperm(len(dataset))
    labeled_idx = []
    # Go over the dataset to select same number of labeled samples from each of the 10 classes
    for i in range (10):
        for j in range(len(dataset)):
            if dataset[j][1] == i and len(labeled_idx) < (i+1) * labeled_len // 10:
                labeled_idx.append(j)
                if len(labeled_idx) // (i+1) == (labeled_len // 10):
                    break
    
    # Separate the unlabeled examples
    unlabeled_idx = list(set(indexer.tolist())-set(labeled_idx))                
    
    print("Number of labeled images:",len(labeled_idx),"unlabeled images:",len(unlabeled_idx))
    
    labeled_data = torch.zeros(labeled_len,1,28,28)
    labeled_labels = torch.zeros(labeled_len,dtype=torch.int64)
    
    unlabeled_data = torch.zeros(len(unlabeled_idx),1,28,28)
    unlabeled_labels = torch.zeros(len(unlabeled_idx),dtype=torch.int64)
    
    i = 0
    
    for idx in labeled_idx:
        labeled_data[i] = dataset[idx][0]
        labeled_labels[i] = dataset[idx][1]
        i += 1
        
    i = 0  
    for idx in unlabeled_idx:
        unlabeled_data[i] = dataset[idx][0]
        unlabeled_labels[i] = dataset[idx][1]
        i += 1
    
    # Create the dataset objects for unlabeled and labeled partitions
    labeled_ds = MyDataset(labeled_data,labeled_labels,transform)
    unlabeled_ds = MyDataset(unlabeled_data,unlabeled_labels,transform)

    return labeled_ds, unlabeled_ds
