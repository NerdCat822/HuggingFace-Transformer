from transformers import AutoImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)

#wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

import h5py

with h5py.File('nyu_depth_v2_labeled.mat', 'r') as file:
  print(file["images"])
  print(file["depths"])

import torch
import h5py
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from transformers import GLPNForDepthEstimation, GLPNImageProcessor
import torch.nn.functional as F

# Load the dataset
with h5py.File('nyu_depth_v2_labeled.mat', 'r') as file:
    images = torch.tensor(file['images'][()]).permute(0, 2, 3, 1)  # Convert to PyTorch format
    depths = torch.tensor(file['depths'][()]).unsqueeze(1)  # Convert to PyTorch format
dataset = TensorDataset(images.float(), depths.float())
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load the model and image processor
image_processor = GLPNImageProcessor.from_pretrained('vinvino02/glpn-kitti')
model = GLPNForDepthEstimation.from_pretrained('vinvino02/glpn-kitti')

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Fine-tune the model
for epoch in range(2):  # Number of epochs
    for i, (images, depths) in enumerate(dataloader):
        # Prepare the inputs
        images = images.to(device)
        depths = depths.to(device)
        inputs = image_processor(images=images.to(device), return_tensors='pt').to(device)

        # Forward pass
        outputs = model(**inputs)

        # Compute the loss
        loss = loss_fn(outputs.predicted_depth, depths)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 100 steps
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/10], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')