from datasets import load_dataset
import random
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the AG News dataset from the Hugging Face library
dataset = load_dataset("ag_news")

# Select a subset of the dataset
subset_size = 10  # Adjust this value according to your desired subset size
random.seed(42)
indices = random.sample(range(len(dataset["train"])), subset_size)
subset_dataset = dataset["train"].select(indices)

# Split the subset dataset into train and validation sets
train_dataset, val_dataset = train_test_split(
    subset_dataset, test_size=0.2, random_state=42
)

train_texts, train_labels = train_dataset["text"], train_dataset["label"]
val_texts, val_labels = val_dataset["text"], val_dataset["label"]

# Load the Longformer tokenizer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# Tokenize the input texts
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=4096  # Adjust this value according to your requirements
)
val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=4096  # Adjust this value according to your requirements
)

# Convert the tokenized inputs into PyTorch tensors
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings["input_ids"]),
    torch.tensor(train_encodings["attention_mask"]),
    torch.tensor(train_labels)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings["input_ids"]),
    torch.tensor(val_encodings["attention_mask"]),
    torch.tensor(val_labels)
)

# Define the Longformer model for sequence classification
model = LongformerForSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096", num_labels=4  # Adjust the number of labels according to your task
)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create data loaders for batching the data during training
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Set the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 10  # Adjust this value according to your requirements
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Training loop
for epoch in range(40):  # Adjust the number of epochs according to your requirements
    model.train()
    train_loss = 0
    train_acc = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        train_acc += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)

    # Evaluation loop
    model.eval()
    val_loss = 0
    val_acc = 0
    val_preds = []
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_acc += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    val_preds = torch.tensor(val_preds)

    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {avg_train_acc:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {avg_val_acc:.4f}")
    print("---------------------------------------------------")

# Save the trained model
model.save_pretrained("longformer_document_classification")
tokenizer.save_pretrained("longformer_document_classification")