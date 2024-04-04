import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageOrientationDataset
from model import ImageOrientationModel
import random
import matplotlib.pyplot as plt
import numpy as np


def calculate_mean_std(dataset, num_samples=50):
    pixel_sum = torch.tensor([0.0, 0.0, 0.0])
    pixel_sum_sq = torch.tensor([0.0, 0.0, 0.0])
    num_pixels = 0

    random_indices = random.sample(range(len(dataset)), num_samples)

    for idx in random_indices:
        image, _ = dataset[idx]
        image = transforms.ToTensor()(image)  # Convert PIL Image to tensor
        pixel_sum += image.sum(dim=[1, 2])
        pixel_sum_sq += (image ** 2).sum(dim=[1, 2])
        num_pixels += image.shape[1] * image.shape[2]

    mean = pixel_sum / num_pixels
    std = torch.sqrt((pixel_sum_sq / num_pixels) - (mean ** 2))

    return mean, std



# Create the dataset (without transformations)
dataset = ImageOrientationDataset(image_dir="C:/Users/User/Downloads/image crops augmented")

# Calculate the mean and standard deviation of the dataset
mean, std = calculate_mean_std(dataset)

# Define the data transformations (updated with normalization)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)  # Normalize with calculated mean and std
])

# Create a new dataset with transformations
dataset = ImageOrientationDataset(image_dir="C:/Users/User/Downloads/image crops augmented", transform=transform)
eval_dataset = ImageOrientationDataset(image_dir="C:/Users/User/Downloads/image crops augmented dev set", transform=transform)

# Create the data loader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Create the model
model = ImageOrientationModel()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
print_interval = 10 # print every 10 batches
terminate_interval = 20 # terminate after 20 batches
training_complete = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    if training_complete:
        break

    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(data_loader, 1):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % print_interval == 0:
            average_loss = running_loss / print_interval
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(data_loader)}], Loss: {average_loss:.4f}")
            running_loss = 0.0

        if batch_idx == terminate_interval:
            print(f"Training stopped after {batch_idx} batches.")
            training_complete = True
            break

print("Training completed!")
model_path = "C:/Users/User/PycharmProjects/horizontal or vertical/trained_model_first epoch_20 batches only.pth"
torch.save(model.state_dict(), model_path)
print(f"Trained model saved as {model_path}")

# Evaluate the model
model_path = "C:/Users/User/PycharmProjects/horizontal or vertical/trained_model_first epoch_20 batches only.pth"
model = ImageOrientationModel()
model.load_state_dict(torch.load(model_path))
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in eval_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Display images and predictions
        if total == batch_size:
            num_rows = 4
            num_cols = (batch_size + num_rows - 1) // num_rows

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
            axs = axs.flatten()

            for i in range(batch_size):
                image = images[i].permute(1, 2, 0).cpu().numpy()
                image = std * image + mean  # Denormalize the image
                image = np.clip(image, 0, 1)  # Clip the pixel values to [0, 1]

                label = labels[i].item()
                pred = predicted[i].item()

                axs[i].imshow(image)
                axs[i].set_title(
                    f"True: {'Horizontal' if label == 0 else 'Vertical'}\nPred: {'Horizontal' if pred == 0 else 'Vertical'}")
                axs[i].axis('off')

            plt.tight_layout()
            plt.show()

            break

accuracy = correct / total
print(f"Evaluation Accuracy: {accuracy:.4f}")
