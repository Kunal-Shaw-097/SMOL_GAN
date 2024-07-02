import torch.optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from gan import Generator, Discriminator
import matplotlib.pyplot as plt
import torch.nn.functional as f
import cv2


def draw_sample(z, ii = 1):
    noise = torch.randn(batch_size, z).to(device)
    out = gen(noise)
    
    for i in range(ii):
        outs = out[i]
        #out = out.view(28, -1)
        outs = outs.squeeze(0)
        plt.imshow(outs.detach().cpu().numpy(), cmap='Greys')
        plt.show()


torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    torchvision.datasets.MNIST("dataset/", download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True)

b = 0


lr = 0.0002
z = 64

gen = Generator(z_dim=z).to(device)
beta_1 = 0.5 
beta_2 = 0.999
dis = Discriminator().to(device)

optimizer_gen = torch.optim.Adam(gen.parameters(), lr= lr, betas=(beta_1, beta_2))
optimizer_dis = torch.optim.Adam(dis.parameters(), lr= lr, betas=(beta_1, beta_2))

draw_sample(z)

epochs = 20

for i in range(epochs):

    b = 0

    print(f"Epoch:{i}")

    while b + batch_size  < imgs.shape[0] : 
        optimizer_dis.zero_grad()  
        real = imgs[b: b + batch_size].to(device).float().unsqueeze(1)
        #real = real.view(batch_size, -1)
        real_labels = torch.ones(batch_size, 1).to(device)

        noise = torch.randn(batch_size, z).to(device)
        fake = gen(noise)

        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        fake_pred = dis(fake.detach())
    
        loss1 = f.binary_cross_entropy(fake_pred, fake_labels)
        
        real_preds = dis(real)
        loss2 = f.binary_cross_entropy(real_preds, real_labels)

        loss_d = (loss1 + loss2)/2

        loss_d.backward()
        optimizer_dis.step()


        optimizer_gen.zero_grad() 
        noise = torch.randn(batch_size, z).to(device)
        fake2 = gen(noise)
        fake_pred2 = dis(fake2)

        loss_g = f.binary_cross_entropy(fake_pred2, torch.ones(batch_size, 1).to(device))

        #loss_g = torch.log(1-fake_pred2).mean()
        loss_g.backward()    
        optimizer_gen.step()

        print(loss_d, loss_g)

        b += batch_size


draw_sample(z, 10)






