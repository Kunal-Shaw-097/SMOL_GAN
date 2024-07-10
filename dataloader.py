import torch
import os
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np

def read_imgs(paths : list = [], size = (64, 64)):
    """
    Returns a numpy arrays consisting all the images read from the paths
    """
    imgs  = []
    for path in paths :
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                             # easier to display RGB images on plt
        img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)
        imgs.append(img)
    return np.array(imgs)     


class LSUN(Dataset):
    def __init__(self, img_dir : str, size : int = 64, norm_center : str = 'zero') :
        """
        img_dir : path to real image dirs
        norm_center : control the type of normalization. 'zero' normalizes between [-1, 1] and 'mean' normalizes between [0, 1]
        """
        if  not Path(img_dir).exists :
            print("Image Dir not found. Exiting ....")
            exit()
        assert norm_center in ['zero', 'mean'] , "Invalid normalization"
        self.imgs = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
        self.norm = norm_center
        self.size = size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index) :
        return self.imgs[index] , 1
    
    def collate_fn(self, batch):
        paths , y =zip(*batch)
        x = read_imgs(paths, (self.size, self.size))
        x = torch.from_numpy(x).float()
        x = x/255                                          # normalize from 0 - 255 to 0 - 1 (need to try this directly using sigmoid)
        if self.norm == 'zero' : 
            x = (x - 0.5) /0.5                                 # normailize from 0 - 1 to  -1 - 1 for tanh (DCGAN paper does this)
        x = x.permute(0, 3 , 1, 2).contiguous()
        y = torch.tensor(y).unsqueeze(1).float()
        return x, y
    
class GaussianNoiseAdder():
    def __init__(self,std : float = 0.2, decay_rate : float = 0.1, decay_steps : int = 100, device : str = 'cpu'):
        self.std = std
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.device = device

    def apply(self, x : torch.Tensor, step : int) -> torch.Tensor:
        # x ---> (B, C, H , W)
        if step % self.decay_steps == 0 :
            self.std = self.std * (1 - self.decay_rate)
        noise = torch.randn(x.shape) * self.std
        return x + noise