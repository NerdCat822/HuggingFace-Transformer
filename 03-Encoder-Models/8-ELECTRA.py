import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# Load the AG News dataset from the datasets library
dataset = load_dataset('ag_news')

# # Extract the training and testing splits
# train_dataset = dataset['train']
# test_dataset = dataset['test']

train_dataset = dataset["train"].select(range(2000))
test_dataset = dataset["test"].select(range(1000))

# Set the device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
batch_size = 16
num_epochs = 5
learning_rate = 2e-5
max_length = 128

# Define your own dataset class inheriting from torch.utils.data.Dataset
class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.texts = dataset['text']
        self.labels = dataset['label']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator', num_labels=4)

train_dataset = CustomDataset(train_dataset, tokenizer, max_length)
test_dataset = CustomDataset(test_dataset, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the ELECTRA model
model.to(device)

# Define the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
    print(f"Loss is {loss}")
# Evaluation loop
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        _, predicted_labels = torch.max(logits, dim=1)

        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
print('Test Accuracy: {:.2f}%'.format(accuracy * 100))