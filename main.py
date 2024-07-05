import torch
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as f
from torchvision.utils import make_grid

from pathlib import Path
import json

from gan import DcGanX128Discriminator, DcGanX128Generator, DcGanX64Discriminator, DcGanX64Generator , DcGanX32Discriminator, DcGanX32Generator
from dataloader import LSUN

import matplotlib.pyplot as plt
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.cuda.empty_cache()

batch_size = 128

lr = 0.0002
z = 100
beta_1 = 0.5
beta_2 = 0.999

epochs = 5

save_dir = Path("results/")

if __name__=="__main__": 

    dataset = LSUN("LSUN/", size = 128)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6 , collate_fn=dataset.collate_fn)

    gen = DcGanX128Generator(100, 3, 512).to(device)
    dis = DcGanX128Discriminator(3, 512).to(device)

    optimizer_gen = torch.optim.Adam(gen.parameters(), lr= lr, betas=(beta_1, beta_2))
    optimizer_dis = torch.optim.Adam(dis.parameters(), lr= lr, betas=(beta_1, beta_2))

    log_steps = 1                       
    step = 0

    total_gen_loss = 0
    total_dis_loss = 0

    gen_losses = []
    dis_losses = []

    sample_generation_per_epoch = []
    steps_per_epoch = len(dataloader)

    for i in range(epochs):

        pbar = tqdm(dataloader, total=len(dataloader))

        for imgs, y in pbar: 

            # ---------------- step for Discriminator-------------------------#
            optimizer_dis.zero_grad() 

            batch_size = imgs.shape[0]
            real = imgs.to(device)

            real_labels = y.to(device)
            noise = torch.randn(batch_size, z).to(device)

            fake = gen(noise)

            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            fake_pred = dis(fake.detach())
            real_preds = dis(real)

            loss1 = f.binary_cross_entropy(fake_pred, fake_labels)
            loss2 = f.binary_cross_entropy(real_preds, real_labels)

            loss_d = (loss1 + loss2)/2

            loss_d.backward()
            optimizer_dis.step()

            # ---------------- step for Generator -------------------------#

            optimizer_gen.zero_grad() 

            fake_pred2 = dis(fake)
            
            loss_g = f.binary_cross_entropy(fake_pred2, torch.ones(batch_size, 1).to(device))

            loss_g.backward()    
            optimizer_gen.step()

            # -------------------- logging and stuff -------------------------#

            total_gen_loss += loss_g
            total_dis_loss += loss_d

            avg_gen_loss = total_gen_loss / (step + 1)
            avg_dis_loss = total_dis_loss / (step + 1)

            pbar.set_description(f"TRAINING Epoch {i + 1} : AVG dis_loss : {avg_dis_loss:03f}, AVG gen_loss : {avg_gen_loss:03f}")

           

            if step % log_steps == 0 or step == len(dataloader) - 1:
                gen_losses.append((step + 1 ,avg_gen_loss.item()))
                dis_losses.append((step + 1, avg_dis_loss.item()))

            step += 1

            if step %10 == 0 :
                break

        sample_generation_per_epoch.append(fake[ :4, : , : , :].detach().permute(0, 2, 3, 1).cpu().numpy())
    

    # -----------------Saving the model and losses-------------------------------------

    model_ckpt = {
        "Generator" : gen.state_dict(),
        "Discriminator" : dis.state_dict(),
    }

    loss_ckpt = {
        "gen_losses" : gen_losses,
        "dis_losses" : dis_losses,
        "steps_per_epoch" : steps_per_epoch
    }

    if not save_dir.exists() :
        save_dir.mkdir()

    torch.save(model_ckpt, save_dir / "GANmodel.pt")

    with open(save_dir / "losses.json", 'w') as f:
        json.dump(loss_ckpt, f)


    #-----------Visualizations--------------------------------

    def show_tensor_images(image_tensor, num_images=25):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
        '''
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()
    
    show_tensor_images(fake, min(16, batch_size))
    show_tensor_images(real, min(16, batch_size))


    # -------------- Loss Plots and save-----------------------------------
    steps, glosses = zip(*gen_losses)
    steps, dlosses = zip(*dis_losses)

    # Calculate the step indices where each epoch starts    
    epoch_starts = list(range(steps[0], steps[-1] + 1, steps_per_epoch))
    # Creating plots for logged losses
    plt.figure(figsize=(10, 6))
    plt.plot(steps, glosses, marker='o', linestyle='-', color='b', label='Generator Loss')
    plt.plot(steps, dlosses, marker='*', linestyle='-', color='r', label='Discriminator Loss')
    plt.scatter(epoch_starts, [0] * len(epoch_starts), color='g', zorder=5, label="New Epoch start")  
    plt.title('Loss Over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "loss_plot.png", format='png', dpi=300)
    plt.show()

    # ------------------- View generations over epochs and save  -------------------------------
    fig, axes = plt.subplots(epochs, 4, figsize=(8, 3 * epochs))
    for i, epoch_images in enumerate(sample_generation_per_epoch):
        for j, img in enumerate(epoch_images):
            ax = axes[i, j]
            img = (img + 1) / 2
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(f'Epoch {i+1}', loc='left')
    # Adjust spacing
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_generation_plot.png', format='png', dpi=300)
    plt.show()







