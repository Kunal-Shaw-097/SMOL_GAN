import torch
import torch.nn as nn



class Generator(nn.Module) :
    def __init__(self, z_dim : int = 10, hidden_dim : int = 64, im_channel : int = 1) :
        """
        z_dim - noise dimension
        hidden_dim - parameter can tune as needed
        im_channel - number of output channels of the generated image 1 for grayscale and 3 for rgb
        """
        super().__init__()
        self.block = nn.Sequential(
            self.gen_block(z_dim, hidden_dim * 4, 3, 2),             # 1 - 3
            self.gen_block(hidden_dim * 4, hidden_dim * 2, 4, 1),    # 3 - 6
            self.gen_block(hidden_dim * 2, hidden_dim, 3, 2),        # 6 - 13
        )
        self.final= nn.ConvTranspose2d(hidden_dim, im_channel, 4, 2)  # 13 - 28
        self.act = nn.Tanh()                                      # Not sure about this

    def gen_block(self, in_dim, out_dim, kernel_size : int  = 3, stride : int = 1, padding : str|int = 'valid') :
        """
        out_height = strides[0] * (in_height - 1) + kernel_size[0] - 2 * padding_height + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
        out_width  = strides[1] * (in_width - 1) + kernel_size[1] - 2 * padding_width + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size, z_dim = x.shape[0], x.shape[1]
        x= x.view(batch_size, z_dim, 1, 1)
        x = self.block(x)
        x = self.final(x)
        x = self.act(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, in_channels : int = 1, hidden_dim : int = 16):
        super().__init__()
        """
        only works for 28 x 28
        """
        self.block = nn.Sequential(
            self.dis_block(in_channels, hidden_dim, 4, 2),                       
            self.dis_block(hidden_dim, hidden_dim * 2, 4, 2),                          
            self.dis_block(hidden_dim * 2, 1, 4, 2),
        )                
        self.act = nn.Sigmoid()

    def dis_block(self, in_dim, out_dim, kernel_size : int  = 3, stride : int = 1, padding : str|int = 'valid'):
        """
        out_height = (in_height + 2 * padding_height - dilation[0] * (kernel_size[0] - 1) - 1 ) / stride[0]  + 1
        out_width = (in_width + 2 * padding_width - dilation[1] * (kernel_size[1] - 1) - 1 ) / stride[1]  + 1
        """
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.block(x)
        x = x.view(batch_size, -1)
        x = self.act(x)
        return x
    

if __name__=="__main__":

    x = torch.rand(32 , 1, 28, 28)

    dis = Discriminator()
    gen = Generator()

    sum  = 0

    for i in dis.parameters():
        sum += i.numel()
    print(sum)

    sum  = 0

    for i in gen.parameters():
        sum += i.numel()
    print(sum)

    x = dis(x)

    print(x.shape)
        