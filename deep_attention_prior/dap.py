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

# Set a seed for reproducibility
np.random.seed(42)

# Dummy data for illustration (64x64)
data = np.random.rand(100, 3, 64, 64).astype(np.float32)
target = data.copy()

# Define transforms
transform = transforms.Compose([transforms.ToTensor()])

# Create custom dataset and data loader
custom_dataset = CustomDataset(data, transform)
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

import torch.nn.functional as F

class DeepAttentionPrior(nn.Module):
    def __init__(self, k, C, D, N, L):
        super(DeepAttentionPrior, self).__init__()

        # Embedding layer
        self.embedding = nn.Conv2d(in_channels=C, out_channels=64, kernel_size=1).to(device)

        # Stack of Convolution and Self Attention Layers
        self.layers = nn.ModuleList()
        for _ in range(L):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=D, out_channels=D, kernel_size=k),
                SelfAttentionLayer(D)
            ))

        # 1x1 Convolution before Sigmoid
        self.final_conv = nn.Conv2d(in_channels=D, out_channels=64, kernel_size=1).to(device)

    def forward(self, Y_tilde):
        # Initial embedding
        embedded = self.embedding(Y_tilde)

        # Stack of Convolution and Self Attention Layers
        for layer in self.layers:
            embedded = layer(embedded)

        # 1x1 Convolution before Sigmoid
        output = self.final_conv(embedded)
        output = F.sigmoid(output)

        # Upsample output to match the input size (64x64)
        output = F.interpolate(output, size=Y_tilde.shape[-2:], mode='bilinear', align_corners=False)

        return output

class SelfAttentionLayer(nn.Module):
    def __init__(self, D):
        super(SelfAttentionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=D, out_channels=D, kernel_size=1).to(device)
        self.similarity_matrix = nn.Parameter(torch.randn(D, D).to(device))

    def forward(self, x):
        q = self.conv(x).view(x.size(0), -1)
        k = q
        v = x.view(x.size(0), -1)

        attn_weights = F.softmax(torch.mm(q, k.t()), dim=-1)
        attn_output = torch.mm(attn_weights, v).view(x.size())

        return attn_output




class InverseImagingModel(nn.Module):
    def __init__(self, k, C, D, N, L):
        super(InverseImagingModel, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=C, out_channels=D, kernel_size=1).to(device)
        self.fc_layer = nn.Conv2d(in_channels=D, out_channels=C, kernel_size=1).to(device)
        self.output_layer = nn.Conv2d(in_channels=C, out_channels=3, kernel_size=1).to(device)
        self.N = N
        self.similarity_matrix = nn.Parameter(torch.randn(C, C).to(device))
        self.feature_maps = []
        self.conv_layer.register_forward_hook(self.hook_fn)

        # Integrate DeepAttentionPrior
        self.dap = DeepAttentionPrior(k, C, D, N, L)

    def hook_fn(self, module, input, output):
        self.feature_maps.append(output)

    def forward(self, Y_tilde):
        batch_size = Y_tilde.size(0)
        # Set a random seed for reproducibility
        torch.manual_seed(42)
        mask = torch.bernoulli(0.5 * torch.ones_like(Y_tilde))  # Generate mask images with probability p=0.5
        Y_tilde_mod = Y_tilde.view(batch_size, C, 64, 64)
        conv_output = self.conv_layer(Y_tilde_mod)
        g_output = torch.relu(conv_output)
        fc_output = self.fc_layer(g_output)

        # Deep attention prior
        dap_output = self.dap(fc_output)

        # With this modified line
        linear_transform = nn.Conv2d(in_channels=D, out_channels=C, kernel_size=1).to(device)
        attn_output_transformed = linear_transform(dap_output.permute(0, 2, 1, 3))
        Y_hat = fc_output + attn_output_transformed

        if fc_output.shape != mask.shape:
            Y_hat = Y_hat.permute(0, 2, 1, 3)
            Y_hat = Y_hat * mask
            Y_hat = Y_hat.permute(0, 2, 1, 3)

        # Reshape fc_output for matrix multiplication
        fc_output = Y_hat.view(batch_size, 64 * 64, C)
        fc_output = torch.matmul(fc_output, self.similarity_matrix)
        fc_output = fc_output.view(batch_size, C, 64, 64)  # Adjust dimensions to match the input
        output = self.output_layer(fc_output)
        output = output.permute(0, 2, 1, 3)

        return output, Y_hat


#####################


# Instantiate the model and send it to the GPU
k = 3  # Kernel size
C = 3  # Number of channels in the image
D = 64  # Number of kernels
N = data.shape[0]  # Number of instances in the dataset
L = 4 # Number of layers
model = InverseImagingModel(k, C, D, N, L).to(device)




# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20

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
    image = cv2.resize(image, (64, 64))
    image = image.transpose(2, 0, 1)
    image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0
    input_images.append(image_tensor)

input_images = torch.stack(input_images, dim=0).to(device)  # Create a batch of input images

# Visualize input and output images for the batch
visualize_input_output(model, input_images)
