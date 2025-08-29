import cv2
import torch
import torchvision
from PIL import Image
from torch import optim
from torchvision.transforms import transforms
from src.Genenator import Generator
from src.Discriminator import Discriminator

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Change path test image
image_path = "/home/loc/Documents/Painting-Images-With-GAN/data/Test/real_333.jpg"

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
BATCH_SIZE = 128
# Hyperparameters


# Initialize networks

generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999))
start_epoch = 0

generator, optimizer_G, start_epoch = load_checkpoint(generator, optimizer_G, filename = "/home/loc/Documents/Painting-Images-With-GAN/models/checkpointG.pth.tar")
generator = generator.to(DEVICE)
image = Image.open(image_path).convert('RGB')
image = image.convert('L')
image = image.convert('RGB')
image = transform(image)

image = image.unsqueeze(0)
image = image.to(DEVICE)

generator.eval()
output = generator(image)
print(start_epoch)
torchvision.utils.save_image(output * 0.5 + 0.5, f"/home/loc/Documents/Painting-Images-With-GAN/images/test.png")