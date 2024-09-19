import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class dataset(Dataset):
  def __init__(self, batch_size, img_size, images_paths, targets, kind):
    self.batch_size = batch_size
    self.img_size = img_size
    self.images_paths = images_paths
    self.targets = targets
    self.kind = kind
    self.len = len(self.images_paths) // batch_size

    self.transform_img_rgb = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),])

    self.normalize_swir = transforms.Normalize(mean=[0.0, 0.0], std=[1.0, 1.0])


    self.batch_im = [self.images_paths[idx * self.batch_size:(idx + 1) * self.batch_size] for idx in range(self.len)]
    self.batch_t = [self.targets[idx * self.batch_size:(idx + 1) * self.batch_size] for idx in range(self.len)]

  def __getitem__(self, idx):
      if self.kind == 'RGB':
        pred = torch.stack([
                self.transform_img_rgb(Image.open(f'../data/S2_dataset/10/{self.kind}/'+file_name))
                for file_name in self.batch_im[idx]
            ])
      elif self.kind == 'SWIR':
        pred = torch.stack([
                self.transform_swir_image(np.load(f'../data/S2_dataset/10/{self.kind}/'+file_name))
                for file_name in self.batch_im[idx]
            ])
      else:
        raise ValueError(f"Unsupported image type: {self.kind}")

      target = torch.tensor(self.batch_t[idx]).float()

      return pred, target

  def __len__(self):
      return self.len

  def transform_swir_image(self, np_image):
        # np_image is (2, 502, 502)
        # Convert each channel to PIL, resize, and convert back to tensor
        c, h, w = np_image.shape
        resized_channels = []
        for i in range(c):
            channel = np_image[i, :, :]
            pil_image = Image.fromarray(channel)
            pil_image = pil_image.resize(self.img_size, Image.LANCZOS)
            resized_channels.append(torch.from_numpy(np.array(pil_image, dtype=np.float32) ))

        # Stack channels and permute to get (C, H, W)
        resized_image = torch.stack(resized_channels).permute(0, 1, 2)

        # Normalize
        normalized_image = self.normalize_swir(resized_image)

        return normalized_image
