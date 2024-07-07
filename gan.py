import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A basic Conv block with a
    -> Conv2d
    -> BatchNorm
    -> LeakyReLU
    """
    def __init__(self, in_dim : int, out_dim : int, kernel_size : int, stride : int, padding : int|str = 'valid', conv_only : bool = False, norm_affine : bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias= False)
        self.conv_only = conv_only
        if not self.conv_only : 
            self.norm = nn.BatchNorm2d(out_dim, affine= norm_affine)       # affine = false, sets scale to 1 and bias to 0 as suggested by the paper
            self.act = nn.LeakyReLU(0.2)                                   #DCGAN paper uses 0.2 for leakyrelu

    def forward(self, x : torch.Tensor):
        x = self.conv(x)
        if not self.conv_only : 
            x = self.norm(x)
            x = self.act(x)
        return x
    
class ConvTransposeBlock(nn.Module):
    """
    A basic Conv Transpose block with a
    -> ConvTranspose2d
    -> BatchNorm
    -> ReLU
    """
    def __init__(self, in_dim : int, out_dim : int, kernel_size : int , stride : int, padding : int = 0, out_padding : int = 0, convt_only : bool = False, norm_affine : bool = False):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, out_padding, bias= False)
        self.convt_only = convt_only
        if not self.convt_only :
            self.norm = nn.BatchNorm2d(out_dim, affine= norm_affine)        # affine = false, sets scale to 1 and bias to 0 as suggested by the paper
            self.act = nn.ReLU()

    def forward(self, x : torch.Tensor):
        x = self.convt(x)
        if not self.convt_only :
            x = self.norm(x)
            x = self.act(x)
        return x
    

# class LinearUpscaler(nn.Module):
#     def __init__(self, in_dim : int = 3, out_dim : int = 1024, norm_affine : bool = False):
#         super().__init__()
#         self.linear = nn.Linear(in_dim, 16 * out_dim, bias= False)
#         self.norm = nn.BatchNorm2d(out_dim, affine=norm_affine)
#         self.act = nn.ReLU()

#     def forward(self, x : torch.Tensor):
#         batch_size = x.shape[0]
#         x = self.linear(x)                                                # (B, z_dim) -----> (B, 16 * hdim)
#         x = x.view(batch_size, -1, 4, 4)                                  # (B, 16 * hdim) -----> (B, hdim, 4 ,4)
#         x = self.norm(x)
#         x = self.act(x)
#         return x    

class BaseGenerator(nn.Module):
    """
    Not to be called directly
    It is a base class for Generators of various input sizes
    """
    def __init__(self) :
        super().__init__()
        self.gen_block = None
        # TODO : try using sigmoid, maybe ?
        self.act = nn.Tanh()                         

    def _init_weights(self, module):
        if isinstance(module , nn.ConvTranspose2d) or isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)   # DCGAN paper suggest this weight initialization
            if module.bias is not None :
                torch.nn.init.zeros_(module.bias) 
        elif isinstance(module, nn.BatchNorm2d):
            if module.weight is not None :
                torch.nn.init.normal_(module.weight, mean=1, std=0.02)
            if module.bias is not None :
                torch.nn.init.zeros_(module.bias)

    def forward(self, x : torch.Tensor):
        x = x.unsqueeze(-1).unsqueeze(-1)                                  # from (B, z_dim) to (B, z_dim , 1, 1), basically a 1 x 1 image
        x = self.gen_block(x)
        x = self.act(x)
        return x
    
    
class BaseDiscriminator(nn.Module):
    """
    Not to be called directly
    It is a base class for Discriminators of various input sizes
    """
    def __init__(self):
        super().__init__()
        self.dis_block = None
        self.act = nn.Sigmoid()

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) :
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)  # DCGAN paper suggest this weight initialization
            if module.bias is not None :
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                torch.nn.init.normal_(module.weight, mean=1, std=0.02)
            if module.bias is not None :
                torch.nn.init.zeros_(module.bias)

    def forward(self, x : torch.Tensor):
        x = self.dis_block(x)
        x = x.squeeze(-1).squeeze(-1)                                     # from (B, 1, 1, 1) to (B , 1), probabiltiy prediciton for the batch 
        x = self.act(x)
        return x
    
class DcGanX128Generator(BaseGenerator):
    """
    Generator for 128 x 128 images
    """
    def __init__(self, z_dim : int = 100, im_channel : int = 3, hidden_dim : int = 1024, norm_affine : bool = False):
        super().__init__()
        self.gen_block = nn.Sequential(
            ConvTransposeBlock(z_dim, hidden_dim, 4, 1, norm_affine=norm_affine),                                    #  1 ---> 4
            ConvTransposeBlock(hidden_dim, hidden_dim//2, 4, 2, 1, norm_affine=norm_affine),                         #  4 ---> 8
            ConvTransposeBlock(hidden_dim//2, hidden_dim//4, 4, 2, 1, norm_affine=norm_affine),                      #  8 ---> 16
            ConvTransposeBlock(hidden_dim//4, hidden_dim//8, 4, 2, 1, norm_affine=norm_affine),                      # 16 ---> 32
            ConvTransposeBlock(hidden_dim//8, hidden_dim//16, 4, 2, 1, norm_affine=norm_affine),                     # 32 ---> 64
            ConvTransposeBlock(hidden_dim//16, im_channel, 4, 2, 1, convt_only=True, norm_affine=norm_affine),       # 64 ---> 128
        )
        self.apply(self._init_weights)


class DcGanX128Discriminator(BaseDiscriminator):
    """
    Discriminator for 128 x 128 images
    """
    def __init__(self, im_channel : int= 3, hidden_dim: int = 1024, norm_affine : bool = False):
        super().__init__()
        self.dis_block = nn.Sequential(
            ConvBlock(im_channel, hidden_dim//16, 4, 2, 1, conv_only=True, norm_affine=norm_affine),             #128 ----> 64
            nn.LeakyReLU(0.2),                                                                                   # DCGAN paper suggested not to use Batchnorm in first block    
            ConvBlock(hidden_dim//16, hidden_dim//8, 4, 2, 1, norm_affine=norm_affine),                          # 64 ----> 32
            ConvBlock(hidden_dim//8, hidden_dim//4, 4, 2, 1, norm_affine=norm_affine),                           # 32 ----> 16
            ConvBlock(hidden_dim//4, hidden_dim//2, 4, 2, 1, norm_affine=norm_affine),                           # 16 ----> 8
            ConvBlock(hidden_dim//2, hidden_dim, 4, 2, 1, norm_affine=norm_affine),                              #  8 ----> 4
            ConvBlock(hidden_dim , 1, 4, 1, conv_only= True, norm_affine=norm_affine),                           #  4 ----> 1
        )
        self.apply(self._init_weights)


class DcGanX64Generator(BaseGenerator):
    """
    Generator for 64 x 64 images
    """
    def __init__(self, z_dim : int = 100, im_channel : int = 3, hidden_dim : int = 1024, norm_affine : bool = False):
        super().__init__()
        self.gen_block = nn.Sequential(
            ConvTransposeBlock(z_dim, hidden_dim, 4, 1, norm_affine=norm_affine),                                    #  1 ---> 4
            ConvTransposeBlock(hidden_dim, hidden_dim//2, 4, 2, 1, norm_affine=norm_affine),                         #  4 ---> 8
            ConvTransposeBlock(hidden_dim//2, hidden_dim//4, 4, 2, 1, norm_affine=norm_affine),                      #  8 ---> 16
            ConvTransposeBlock(hidden_dim//4, hidden_dim//8, 4, 2, 1, norm_affine=norm_affine),                      # 16 ---> 32
            ConvTransposeBlock(hidden_dim//8, im_channel, 4, 2, 1, convt_only=True, norm_affine=norm_affine),        # 32 ---> 64
        )
        self.apply(self._init_weights)
    

class DcGanX64Discriminator(BaseDiscriminator):
    """
    Discriminator for 6 x 64 images
    """
    def __init__(self, im_channel : int= 3, hidden_dim: int = 1024, norm_affine : bool = False):
        super().__init__()
        self.dis_block = nn.Sequential(
            ConvBlock(im_channel, hidden_dim//8, 4, 2, 1, conv_only=True, norm_affine=norm_affine),              # 64 ----> 32
            nn.LeakyReLU(0.2),
            ConvBlock(hidden_dim//8, hidden_dim//4, 4, 2, 1, norm_affine=norm_affine),                           # 32 ----> 16
            ConvBlock(hidden_dim//4, hidden_dim//2, 4, 2, 1, norm_affine=norm_affine),                           # 16 ----> 8
            ConvBlock(hidden_dim//2, hidden_dim, 4, 2, 1, norm_affine=norm_affine),                              #  8 ----> 4
            ConvBlock(hidden_dim , 1, 4, 1, conv_only= True, norm_affine=norm_affine),                           #  4 ----> 1
        )
        self.apply(self._init_weights)
    

class DcGanX32Generator(BaseGenerator):
    """
    Generator for 32 x 32 images
    """
    def __init__(self, z_dim : int = 100, im_channel : int = 3, hidden_dim : int = 512, norm_affine : bool = False):
        super().__init__()
        self.gen_block = nn.Sequential(
            ConvTransposeBlock(z_dim, hidden_dim, 4, 1, norm_affine=norm_affine),                                       #  1 ---> 4
            ConvTransposeBlock(hidden_dim, hidden_dim//2, 4, 2, 1, norm_affine=norm_affine),                            #  4 ---> 8
            ConvTransposeBlock(hidden_dim//2, hidden_dim//4, 4, 2, 1, norm_affine=norm_affine),                         #  8 ---> 16
            ConvTransposeBlock(hidden_dim//4, im_channel, 4, 2, 1, convt_only= True, norm_affine=norm_affine),          # 16 ---> 32
        )
        self.apply(self._init_weights)
    

class DcGanX32Discriminator(BaseDiscriminator):
    """
    Discriminator for 32 x 32 images
    """
    def __init__(self, im_channel : int = 3, hidden_dim : int = 512, norm_affine : bool = False) :
        super().__init__()
        self.dis_block = nn.Sequential( 
            ConvBlock(im_channel, hidden_dim//4, 4, 2, 1, norm_affine=norm_affine, conv_only= True),             # 32 ----> 16
            nn.LeakyReLU(0.2),                                                                                    
            ConvBlock(hidden_dim//4, hidden_dim//2, 4, 2, 1, norm_affine=norm_affine),                           # 16 ----> 8
            ConvBlock(hidden_dim//2, hidden_dim, 4, 2, 1, norm_affine=norm_affine),                              #  8 ----> 4
            ConvBlock(hidden_dim , 1, 4, 1, conv_only= True, norm_affine=norm_affine),                           #  4 ----> 1
        )
        self.apply(self._init_weights)
    

if __name__=="__main__":
    """
    testing models
    """
    torch.cuda.random.manual_seed(42)
    torch.manual_seed(42)
    torch.random.manual_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    dis_in = torch.randn(32 , 3, 128, 128).to(device)
    gen_in = torch.randn(32,100).to(device)
    
    # Linearup = LinearUpscaler(100, 1024)

    # print(Linearup(gen_in).shape)

    gen = DcGanX128Generator(100, 3, 1024).to(device)
    dis = DcGanX128Discriminator(3,1024).to(device)

    x = gen(gen_in)
    y = dis(dis_in)

    print(x.shape)  
    print(y.shape)    