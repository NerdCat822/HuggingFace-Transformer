from datasets import load_dataset
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Load pre-trained Pegasus model and tokenizer
model_name = 'google/pegasus-cnn_dailymail'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Load and preprocess the dataset
train_dataset = load_dataset('cnn_dailymail', '3.0.0', split='train[:50]')  # Load a subset of the "cnn_dailymail" dataset

def preprocess_function(examples):
    inputs = tokenizer(examples['article'], truncation=True, padding='longest')
    targets = tokenizer(examples['highlights'], truncation=True, padding='longest')
    inputs['labels'] = targets['input_ids']
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=1,
    predict_with_generate=True,
    num_train_epochs=1,
    save_steps=500,
    save_total_limit=3,
    eval_steps=100,
    logging_steps=100,
    logging_dir='./logs',
    learning_rate=1e-5,
    warmup_steps=500,
    weight_decay=0.01,
)

# Create the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
)

# Start training
trainer.train()

model.save_pretrained("path/to/save/directory")

from transformers import PegasusForConditionalGeneration

model = PegasusForConditionalGeneration.from_pretrained("path/to/save/directory")

input_text = ["Your input text goes here"]
input_encoding = tokenizer(input_text, truncation=True, padding="longest", max_length=512, return_tensors="pt")

# Generate predictions
output = model.generate(input_encoding["input_ids"])
predicted_text = tokenizer.batch_decode(output, skip_special_tokens=True)