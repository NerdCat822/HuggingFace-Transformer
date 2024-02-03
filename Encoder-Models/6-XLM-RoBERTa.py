import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Load the tokenizer and pre-trained model
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

# Load the IMDb dataset
dataset = load_dataset("imdb")

subset = dataset["train"].select(range(2000))

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(subset, test_size=0.2, random_state=42)

# Tokenize and encode the input sequences
train_encodings = tokenizer(
    train_data["text"],
    truncation=True,
    padding=True,
    max_length=128
)
val_encodings = tokenizer(
    val_data["text"],
    truncation=True,
    padding=True,
    max_length=128
)

# Convert the labels to tensors
train_labels = torch.tensor(train_data["label"])
val_labels = torch.tensor(val_data["label"])

# Create a PyTorch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

num_epochs = 5

# Prepare optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * num_epochs
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / len(val_dataset)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

# Save the fine-tuned model
model.save_pretrained("path/to/save/model")
tokenizer.save_pretrained("path/to/save/tokenizer")

from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

model = XLMRobertaForSequenceClassification.from_pretrained("/content/path/to/save/model")
tokenizer = XLMRobertaTokenizer.from_pretrained("path/to/save/tokenizer")

text = "This is an example sentence."
inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits
predicted_label = torch.argmax(logits, dim=1).item()

print(predicted_label)
print(val_data["label"][0])