from transformers import MT5ForConditionalGeneration, MT5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('cnn_dailymail', '3.0.0', split='train[:100]')  # Load a subset of the "cnn_dailymail" dataset
# train_dataset = dataset['train']

# Load the mT5 tokenizer
tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')

# Preprocessing function
def preprocess_function(examples):
    inputs = examples['article']
    targets = examples['highlights']
    inputs = [input_text for input_text in inputs]
    targets = [target_text for target_text in targets]
    inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(targets, padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = targets['input_ids']
    return inputs

train_dataset = dataset.map(preprocess_function, batched=True)

model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./output_dir',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=3,
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    tokenizer=tokenizer,
)

# Start training
trainer.train()

model.save_pretrained('./exported_model')
tokenizer.save_pretrained('./exported_tokenizer')

 # Load the trained model
model = MT5ForConditionalGeneration.from_pretrained('/content/exported_model')

# Load the tokenizer
tokenizer = MT5Tokenizer.from_pretrained('/content/exported_tokenizer')