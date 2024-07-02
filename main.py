import torch.optim
import torchvision
import torch
from gan import Generator, Discriminator
import matplotlib.pyplot as plt
import torch.nn.functional as f
import cv2


def draw_sample(z):
    noise = torch.rand(batch_size, z).to(device)
    out = gen(noise)
    out = out[0]
    out = out.view(28, -1)
    print(out.shape)
    plt.imshow(out.detach().cpu().numpy(), cmap='Greys')
    plt.show()


torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = torchvision.datasets.MNIST("dataset/", download= True)

batch_size = 512


imgs = data.data

imgs = imgs/255

plt.imshow(imgs[0].numpy(), cmap='Greys')
plt.show()


b = 0




lr= 0.00001
z = 64

gen = Generator(z_dim=z).to(device)
dis = Discriminator().to(device)

optimizer_gen = torch.optim.Adam(gen.parameters(), lr= lr)
optimizer_dis = torch.optim.Adam(dis.parameters(), lr= lr)




draw_sample(z)

epochs = 10

for i in range(epochs):

    b = 0

    while b + batch_size  < imgs.shape[0] : 
        real = imgs[b: b + batch_size].to(device).float()
        real = real.view(batch_size, -1)
        real_labels = torch.ones(batch_size, 1).to(device)

        noise = torch.rand(batch_size, z).to(device)
        fake = gen(noise)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        fake_pred = dis(fake.detach())
        loss1 = f.binary_cross_entropy(fake_pred, fake_labels)
        
        real_preds = dis(real)
        loss2 = f.binary_cross_entropy(real_preds, real_labels)

        loss_d = (loss1 + loss2)/2

        optimizer_dis.zero_grad()  
        loss_d.backward()
        optimizer_dis.step()

        noise = torch.rand(batch_size, z).to(device)
        fake2 = gen(noise)
        fake_pred2 = dis(fake2)

        loss_g = f.binary_cross_entropy(fake_pred2, torch.ones(batch_size, 1).to(device))

        optimizer_gen.zero_grad() 
        loss_g.backward()    
        optimizer_gen.step()

        print(loss_d, loss_g)

        b += batch_size


draw_sample(z)






