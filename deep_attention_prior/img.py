# import tensorflow as tf
#
# # Load the CIFAR dataset
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
#
# # Find an image with the desired dimensions
# for image in train_images:
#     if image.shape == (32, 32, 3):
#         # Save the image to disk
#         tf.keras.preprocessing.image.save_img('image.png', image)
#         break


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

# Dummy data for illustration
data = np.random.rand(100, 3, 32, 32).astype(np.float32)
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
        Y_tilde = Y_tilde.view(batch_size, C, 32, 32)
        conv_output = self.conv_layer(Y_tilde)
        g_output = torch.relu(conv_output)
        fc_output = self.fc_layer(g_output)

        # Reshape fc_output for matrix multiplication
        fc_output = fc_output.view(batch_size, 32 * 32, C)
        fc_output = torch.matmul(fc_output, self.similarity_matrix)
        fc_output = fc_output.view(batch_size, C, 32, 32)  # Adjust dimensions to match the input
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
num_epochs = 100

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

# Visualize feature maps
# def visualize_feature_maps(model, input_image):
#     model.eval()
#     feature_maps = model.feature_maps
#     with torch.no_grad():
#         model(input_image)
#
#     for i, feature_map in enumerate(feature_maps):
#         plt.figure(figsize=(6, 6))
#         num_features = feature_map.size(1)
#         for j in range(num_features):
#             plt.subplot(num_features // 4, 4, j + 1)
#             plt.imshow(feature_map[0, j].cpu(), cmap='viridis')
#             plt.axis('off')
#         plt.show()
#
# # Create an input image for visualization (replace with your own image)
# input_image = torch.randn(1, 3, 32, 32).to(device)  # Send the input image to the GPU
#
# # Visualize feature maps
# visualize_feature_maps(model, input_image)
