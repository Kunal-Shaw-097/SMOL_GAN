import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim = 64, hidden_dim = 128) -> None:
        super().__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.linear4 = nn.Linear(hidden_dim * 4, hidden_dim * 8)
        self.last = nn.Linear(hidden_dim*8, 784)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x : torch.Tensor):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.act(self.linear3(x))
        x = self.act(self.linear4(x))
        x = self.last(x)
        x = self.sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, hidden_dim = 128) -> None:
        super().__init__()
        self.linear1 = nn.Linear(784, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.act = nn.LeakyReLU(0.2)
        self.pred = nn.Linear(hidden_dim*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x : torch.Tensor): 
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.pred(x)
        x =  self.sigmoid(x)
        return x
    
