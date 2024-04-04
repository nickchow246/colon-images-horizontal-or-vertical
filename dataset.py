import os
from torch.utils.data import Dataset
from PIL import Image

class ImageOrientationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.labels = self.assign_labels()

    def assign_labels(self):
        labels = []
        for image_file in self.image_files:
            if image_file.endswith("rot90.png") or image_file.endswith("rot270.png"):
                labels.append(0)  # Horizontal
            else:
                labels.append(1)  # Vertical
        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(image_path).convert("RGB")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
