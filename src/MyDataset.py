import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        label_image  = image
        gray_img = image.convert('L')
        gray_img_rgb = gray_img.convert('RGB')
        if self.transform:
            input_image = self.transform(gray_img_rgb)
        if self.transform:
            label_image = self.transform(label_image)
        return input_image, label_image
