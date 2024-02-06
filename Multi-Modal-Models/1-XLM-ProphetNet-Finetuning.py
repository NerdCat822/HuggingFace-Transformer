import torch
from transformers import XLMProphetNetTokenizer, XLMProphetNetForCausalLM
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# Define your custom dataset
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
# Load the XLM-ProphetNet model and tokenizer
model_name = 'microsoft/xprophetnet-large-wiki100-cased'
model = XLMProphetNetForCausalLM.from_pretrained(model_name)
tokenizer = XLMProphetNetTokenizer.from_pretrained(model_name)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define hyperparameters
batch_size = 1
max_length = 128
num_epochs = 3
learning_rate = 2e-5

# Load the dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:10]')  # Load the first 100 examples from the WikiText-2 dataset

# Extract texts from the dataset
texts = dataset['text']

# Create the custom dataset
custom_dataset = CustomDataset(texts, tokenizer, max_length)

# Create the data loader
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(data_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Inference
model.eval()
example_prompt = 'Once upon a time'
input_ids = tokenizer.encode(example_prompt, return_tensors='pt').to(device)

with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f'Example Prompt: {example_prompt}')
    print(f'Generated Text: {generated_text}')