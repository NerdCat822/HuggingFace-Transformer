import torch
import torchvision.transforms as transforms
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load the CIFAR-10 dataset
dataset = load_dataset("cifar10")

# Define the subset size
subset_size = 1000

# Take a subset of the dataset
dataset = dataset["train"].select(range(subset_size))

# Define the ViT model
config = ViTConfig.from_pretrained("google/vit-large-patch32-384")
model = ViTForImageClassification(config)

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the batch size and number of training epochs
batch_size = 8
num_epochs = 10

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize the images
    transforms.ToTensor(),
])

# Custom collate function to handle PIL images
def custom_collate(batch):
    images = [transform(item["image"]) for item in batch]
    labels = [item["label"] for item in batch]
    return {"image": torch.stack(images), "label": torch.tensor(labels)}

# Preprocess the dataset
preprocessed_dataset = dataset.map(lambda x: {'image': x['img'], 'label': x['label']})

# Create a data loader for the preprocessed dataset with custom collate function
dataloader = DataLoader(preprocessed_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

print(len(dataloader))

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        # Move batch to device
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(inputs)
        logits = outputs.logits

        # Calculate loss
        loss = loss_fn(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")