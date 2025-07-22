# Author: William Kandolo
import torch
from torch.utils.data import DataLoader, TensorDataset
from generator import Generator
from discriminator import Discriminator
import numpy as np

def train_madgan(generator, discriminator, data_loader, epochs=10):
    gen_opt = torch.optim.Adam(generator.parameters(), lr=1e-4)
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for real in data_loader:
            real = real[0].float()
            batch_size = real.size(0)
            noise = torch.randn(batch_size, real.size(1), real.size(2))

            fake = generator(noise).detach()
            dis_real = discriminator(real)
            dis_fake = discriminator(fake)
            loss_dis = loss_fn(dis_real, torch.ones_like(dis_real)) +                        loss_fn(dis_fake, torch.zeros_like(dis_fake))
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()

            fake = generator(noise)
            dis_fake = discriminator(fake)
            loss_gen = loss_fn(dis_fake, torch.ones_like(dis_fake))
            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()

        print(f"Epoch {epoch}: D_loss={loss_dis.item():.4f} G_loss={loss_gen.item():.4f}")

def compute_anomaly_score(generator, discriminator, input_seq):
    with torch.no_grad():
        noise = torch.randn_like(input_seq)
        generated = generator(noise)
        dis_score = discriminator(input_seq)
        recon_error = torch.mean((input_seq - generated) ** 2, dim=[1, 2])
        return recon_error + (1.0 - torch.sigmoid(dis_score).squeeze())


# Save trained models
torch.save(generator.state_dict(), "models/generator.pth")
torch.save(discriminator.state_dict(), "models/discriminator.pth")
print("Models saved to 'models/' directory.")
