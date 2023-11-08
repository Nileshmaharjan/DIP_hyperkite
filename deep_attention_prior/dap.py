import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# Check if GPU (CUDA) is available and use it, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample.to(device)  # Send the data to the GPU

# Dummy data for illustration (128x128)
data = np.random.rand(100, 3, 128, 128).astype(np.float32)
target = data.copy()

# Define transforms
transform = transforms.Compose([transforms.ToTensor()])

# Create custom dataset and data loader
custom_dataset = CustomDataset(data, transform)
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

class InverseImagingModel(nn.Module):
    def __init__(self, k, C, D, N):
        super(InverseImagingModel, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=C, out_channels=D, kernel_size=1).to(device)
        self.fc_layer = nn.Conv2d(in_channels=D, out_channels=C, kernel_size=1).to(device)
        self.output_layer = nn.Conv2d(in_channels=C, out_channels=3, kernel_size=1).to(device)  # Adjust the output channels here
        self.N = N
        self.similarity_matrix = nn.Parameter(torch.randn(C, C).to(device))
        self.feature_maps = []
        self.conv_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature_maps.append(output)

    def forward(self, Y_tilde):
        batch_size =  Y_tilde.size(0)
        mask = torch.bernoulli(0.5 * torch.ones_like(Y_tilde))  # Generate mask images with probability p=0.5
        Y_tilde_mod = Y_tilde.view(batch_size, C, 128, 128)
        conv_output = self.conv_layer(Y_tilde_mod)
        g_output = torch.relu(conv_output)
        fc_output = self.fc_layer(g_output)
        Y_hat = fc_output

        if fc_output.shape != mask.shape:
            Y_hat = fc_output.permute(0, 2, 1, 3)
            Y_hat = Y_hat * mask
            Y_hat = Y_hat.permute(0, 2, 1, 3)
        #
        # else:
        #     Y_hat = Y_hat * mask


        # Reshape fc_output for matrix multiplication
        fc_output = Y_hat.view(batch_size, 128 * 128, C)
        fc_output = torch.matmul(fc_output, self.similarity_matrix)
        fc_output = fc_output.view(batch_size, C, 128, 128)  # Adjust dimensions to match the input
        output = self.output_layer(fc_output)
        output = output.permute(0, 2, 1, 3)

        return output, Y_hat

# Instantiate the model and send it to the GPU
k = 3  # Kernel size
C = 3  # Number of channels in the image
D = 64  # Number of kernels
N = data.shape[0]  # Number of instances in the dataset
model = InverseImagingModel(k, C, D, N).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 15

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, batch in enumerate(data_loader):
        inputs = batch.to(device)  # Send the input data to the GPU
        optimizer.zero_grad()
        outputs, tilde = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(data_loader)}")

print("Training complete")


import cv2

def visualize_input_output(model, input_images):
    model.eval()
    with torch.no_grad():
        output_images, tildes = model(input_images)

    # Create a figure with 4 subplots
    # Create a figure with 4 subplots for each image in the batch
    fig, axes = plt.subplots(input_images.shape[0], 3, figsize=(12, input_images.shape[0] * 4))

    for i in range(input_images.shape[0]):
        input_image = input_images[i].permute(1, 2, 0).cpu().detach().numpy()
        tilde_image = tildes[i].permute(1, 2, 0).cpu().detach().numpy()
        output_image = output_images[i].permute(0, 2, 1).cpu().detach().numpy()

        # Plot the input image
        axes[i, 0].imshow(input_image)
        axes[i, 0].set_title(f"Reference Image {i + 1}")
        axes[i, 0].axis('off')

        # Plot the tilde image
        axes[i, 1].imshow(tilde_image)
        axes[i, 1].set_title(f"Tilde Input Image {i + 1}")
        axes[i, 1].axis('off')

        # Plot the output image
        axes[i, 2].imshow(output_image)
        axes[i, 2].set_title(f"Output Image {i + 1}")
        axes[i, 2].axis('off')

    plt.show()


# Load and preprocess your images
image_paths = ["attention_1.jpg", "attention_2.jpg", "attention_3.jpg", "attention_4.jpg"]  # Replace with your image paths
input_images = []

for path in image_paths:
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = image.transpose(2, 0, 1)
    image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0
    input_images.append(image_tensor)

input_images = torch.stack(input_images, dim=0).to(device)  # Create a batch of input images

# Visualize input and output images for the batch
visualize_input_output(model, input_images)







