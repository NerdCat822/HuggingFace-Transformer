from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from datasets import load_dataset
import torch

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
model = AutoModelForMaskedLM.from_pretrained('microsoft/codebert-base')

# Load the dataset
dataset = load_dataset("codeparrot/apps")

subset = dataset["train"].select(range(100))

# Tokenize the dataset
def tokenize_data(examples):
    inputs = tokenizer(examples['question'], truncation=True, padding='max_length', max_length=512)
    outputs = tokenizer(examples['solutions'][0], truncation=True, padding='max_length', max_length=512)
    labels = outputs["input_ids"].copy()
    inputs["labels"] = labels
    return inputs

dataset = subset.map(tokenize_data, batched=False)

# Create the DataLoader
def collate_fn(batch):
    keys = ['input_ids', 'attention_mask', 'labels']
    return {key: torch.tensor([item[key] for item in batch]) for key in keys}

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Fine-tuning loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(10):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch}, Loss: {total_loss}")