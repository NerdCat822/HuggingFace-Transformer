from transformers import RobertaForQuestionAnswering, RobertaTokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset

# Load the SQuAD dataset
datasets = load_dataset('squad')

# Select a subset of the dataset
datasets['train'] = datasets['train'].select(range(1000))
datasets['validation'] = datasets['validation'].select(range(100))

# Load the tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Prepare the dataset for training
def prepare_train_features(examples):
    # Some of the questions have no answers, so we need to handle that
    examples['start_positions'] = [answer['answer_start'][0] if len(answer['answer_start']) > 0 else 0 for answer in examples['answers']]
    examples['end_positions'] = [answer['answer_start'][0] + len(answer['text'][0]) if len(answer['answer_start']) > 0 else 0 for answer in examples['answers']]
    return examples

tokenized_datasets = tokenized_datasets.map(prepare_train_features, batched=True)

# Load the model
model = RobertaForQuestionAnswering.from_pretrained('roberta-base')

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    save_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Train the model
trainer.train()

from transformers import pipeline

# Load the trained model and tokenizer
model = RobertaForQuestionAnswering.from_pretrained('/content/results/checkpoint-189')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

# Create a question-answering pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Define the context and the question
context = "The Eiffel Tower is an iron lattice tower located on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel, whose company designed and built the tower."
question = "Who designed the Eiffel Tower?"

# Use the model to answer the question
answer = nlp(question=question, context=context)

print(answer)