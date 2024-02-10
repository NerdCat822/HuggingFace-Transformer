import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
image = download_image(url)

prompt = "turn him into cyborg"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
images[0]

from datasets import load_dataset
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from torchvision import transforms
import numpy as np
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load and preprocess the dataset
dataset = load_dataset('cifar10')
dataset = dataset['train'].select(range(500))  # selecting a subset

def preprocess(data):
    image = data['img']
    image = transform(image)  # Converts PIL Image to PyTorch Tensor
    instruction = 'Make the image look like class ' + str(data['label'])
    return {'image': image, 'instruction': instruction}

dataset = dataset.map(preprocess, remove_columns=['img', 'label'])
dataset.set_format(type='torch', columns=['image', 'instruction'])

# Load the model
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

model_unet = pipe.unet
optimizer = torch.optim.Adam(model_unet.parameters(), lr=0.001)

from torchvision.transforms.functional import to_tensor

def loss_function(output, target):
    output_tensor = to_tensor(output).to(target.device).requires_grad_()
    return ((output_tensor - target) ** 2).mean()

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = pipe.to(device)

# Training loop
for epoch in range(3):  # for 3 epochs
    for data in dataset:
        optimizer.zero_grad()

        # Move data to the device
        image = data['image'].to(device)
        instruction = data['instruction']

        # Forward pass
        outputs = pipe(instruction, image=image, num_inference_steps=10, image_guidance_scale=1)

        # Compute loss
        loss = loss_function(outputs.images[-1], image)  # Now both are tensors
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

print("Training is done.")