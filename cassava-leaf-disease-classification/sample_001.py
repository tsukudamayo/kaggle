
import torch
from torch.utils.data import Dataset, DataLoader


DIR_INPUT ='../../input/cassava-leaf-disease-classification'
DIR_WEIGHTS = '../../input/cassava-pytorch-starter-train'

SEED = 42
N_FOLDS = 1
BATCH_SIZE = 16
SIZE = 512
CROP = 512
init_lr = 5e-5
n_epochs = 5


class CassavaDataset(Dataset):
    def __init__(
        self,
        df,
        dataset='train',
        transforms=None
    ):
        self.df
        self.transforms=transforms
        self.dataset=dataset

    def len(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_src = f'{DIR_INPUT}/{self.dataset}_images/{self.id.loc[idx, "image_id"]}'
        
        
    
        
        
