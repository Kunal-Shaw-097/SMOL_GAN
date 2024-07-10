import torch
from gan import DcGanX64Generator
import matplotlib.pyplot as plt



device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load("results/GANmodel.pt", map_location=device)

model_params = ckpt['model_params']
model_ckpt = ckpt['Generator']


generator = DcGanX64Generator(im_channel= 3, **model_params).to(device)

generator = torch.compile(generator)
for i, modules in ckpt.items():
    print(i)

generator.load_state_dict(model_ckpt)

for j in range(10):

    num_generations = 4

    samples_noises = torch.rand(num_generations, model_params['z_dim']).to(device)

    generations = generator(samples_noises).detach().permute(0, 2, 3, 1).cpu()

    print(torch.min(generations), torch.max(generations))

    fig, ax = plt.subplots(num_generations, 1)
    for i in range(num_generations):
        image = generations[i, : ,: ,:].numpy()
        image = (image * 0.5) + 0.5
        axis = ax[i]
        axis.imshow(image)
        axis.axis('off')
        axis.set_title(f'Generation {i+1}', loc='left')

    plt.tight_layout()
    plt.show()










