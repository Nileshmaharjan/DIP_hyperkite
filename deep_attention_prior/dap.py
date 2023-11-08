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
        batch_size = Y_tilde.size(0)
        Y_tilde = Y_tilde.view(batch_size, C, 128, 128)
        conv_output = self.conv_layer(Y_tilde)
        g_output = torch.relu(conv_output)
        fc_output = self.fc_layer(g_output)

        # Reshape fc_output for matrix multiplication
        fc_output = fc_output.view(batch_size, 128 * 128, C)
        fc_output = torch.matmul(fc_output, self.similarity_matrix)
        fc_output = fc_output.view(batch_size, C, 128, 128)  # Adjust dimensions to match the input
        output = self.output_layer(fc_output)
        output = output.permute(0, 2, 1, 3)

        return output

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
num_epochs = 50

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, batch in enumerate(data_loader):
        inputs = batch.to(device)  # Send the input data to the GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(data_loader)}")

print("Training complete")

# def visualize_feature_maps(model, input_image):
#     model.eval()
#     feature_maps = model.feature_maps
#     with torch.no_grad():
#         model(input_image)
#
#     num_rows, num_cols = 2, 2  # Set the number of rows and columns for the grid
#     total_features = min(num_rows * num_cols, feature_maps[0].size(1))
#
#     plt.figure(figsize=(6, 6))
#     for j in range(total_features):
#         plt.subplot(num_rows, num_cols, j + 1)
#         plt.imshow(feature_maps[0][j, 0].cpu().detach().numpy(), cmap='viridis')  # Access the list and tensor properly
#         plt.axis('off')
#     plt.show()



def visualize_input_output(model, input_image):
    model.eval()
    with torch.no_grad():
        output_image = model(input_image)

    input_image = input_image[0].permute(1, 2, 0).cpu().detach().numpy()
    output_image = output_image[0].permute(0, 2, 1).cpu().detach().numpy()
    print('here')

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title("Output Image")
    plt.axis('off')

    plt.show()

import cv2


# Load your image using OpenCV
image = cv2.imread("attention.jpg")  # Replace with the path to your image

# Resize the image to 128x128 pixels
image = cv2.resize(image, (128, 128))

# Convert the image to the PyTorch tensor format
image = image.transpose(2, 0, 1)  # Change the order of dimensions (H x W x C to C x H x W)
image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
image_tensor = image_tensor.unsqueeze(0)  # Add an extra dimension for batch size

# Now, image_tensor is of size (1, 3, 128, 128)
input_image = image_tensor.to(device)

# Visualize input and output images
visualize_input_output(model, input_image)


# # Create an input image for visualization (replace with your own image)
# input_image = torch.randn(1, 3, 128, 128).to(device)  # Send the input image to the GPU

# # Visualize feature maps in a 2x2 grid
# visualize_feature_maps(model, input_image)

