import os

import torch
import torchvision
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from src.Discriminator import Discriminator
from src.Genenator import Generator
from src.MyDataset import MyDataset

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth.tar"):

    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):

    try:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        return model, optimizer, epoch
    except FileNotFoundError:
        return model, optimizer, 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
BETA1 = 0.5  # For Adam optimizer
LAMBDA_L1 = 100  # Weight for L1 loss
NUM_EPOCHS = 250
BATCH_SIZE = 2

Con = True

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999))
start_epoch = 0

if Con == True:
  generator, optimizer_G, start_epoch = load_checkpoint(generator, optimizer_G, filename = "/home/loc/Documents/Painting-Images-With-GAN/models/checkpointG.pth.tar")
  discriminator, optimizer_D, start_epoch = load_checkpoint(discriminator, optimizer_D, filename = "/home/loc/Documents/Painting-Images-With-GAN/models/checkpointD.pth.tar")
  generator = generator.to(DEVICE)
  discriminator = Discriminator().to(DEVICE)


# Loss functions
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

train_dataset = MyDataset(root_dir='/home/loc/Documents/Painting-Images-With-GAN/data/Train', transform=transform)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
NUM_EPOCHS = 300


for epoch in range(start_epoch, NUM_EPOCHS):
    for batch_idx, (input_img, target_img) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")):
        input_img, target_img = input_img.to(DEVICE), target_img.to(DEVICE)

        discriminator.zero_grad()

        D_real = discriminator(input_img, target_img)
        loss_D_real = criterion_GAN(D_real, torch.ones_like(D_real))

        fake_img = generator(input_img)
        D_fake = discriminator(input_img, fake_img.detach())
        loss_D_fake = criterion_GAN(D_fake, torch.zeros_like(D_fake))

        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        generator.zero_grad()

        D_fake_gen = discriminator(input_img, fake_img)
        loss_G_GAN = criterion_GAN(D_fake_gen, torch.ones_like(D_fake_gen))

        loss_G_L1 = criterion_L1(fake_img, target_img) * LAMBDA_L1

        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optimizer_G.step()

        if batch_idx % 412 == 0:
            with torch.no_grad():
              fake_img = generator(input_img)
              img_grid = torchvision.utils.make_grid(fake_img, normalize=True)
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_loader)} | "
                  f"D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")


            if not os.path.exists("/home/loc/Documents/Painting-Images-With-GAN/images/generated_images"):
                os.makedirs("/home/loc/Documents/Painting-Images-With-GAN/images/generated_images")

            torchvision.utils.save_image(fake_img * 0.5 + 0.5, f"/home/loc/Documents/Painting-Images-With-GAN/images/generated_images/fake_epoch{epoch}_batch{batch_idx}.png")
            torchvision.utils.save_image(input_img * 0.5 + 0.5, f"/home/loc/Documents/Painting-Images-With-GAN/images/generated_images/input_epoch{epoch}_batch{batch_idx}.png")
            torchvision.utils.save_image(target_img * 0.5 + 0.5, f"/home/loc/Documents/Painting-Images-With-GAN/images/generated_images/real_epoch{epoch}_batch{batch_idx}.png")

    #Save models after each epoch or periodically

    save_checkpoint(generator, optimizer_G, epoch, filename = "/home/loc/Documents/Painting-Images-With-GAN/models/checkpointG.pth.tar")
    save_checkpoint(discriminator, optimizer_D, epoch, filename = "/home/loc/Documents/Painting-Images-With-GAN/models/checkpointD.pth.tar")

print("Training finished!")