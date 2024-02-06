from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import AdamW
import torch

# initialize question and context encoders and their tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# here is a random context and a related question
context = "Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals, which involves consciousness and emotionality. The distinction between the former and the latter categories is often revealed by the acronym chosen. 'Strong' AI is usually labelled as AGI (Artificial General Intelligence) while attempts to emulate 'natural' intelligence have been called ABI (Artificial Biological Intelligence)."
question = "What is artificial intelligence?"

# tokenize the input (both question and context)
question_inputs = question_tokenizer(question, return_tensors='pt')
context_inputs = context_tokenizer(context, return_tensors='pt')

# encode the input to get embeddings
question_emb = question_encoder(**question_inputs).pooler_output
context_emb = context_encoder(**context_inputs).pooler_output

# compute the similarity between the embeddings (we can use cosine similarity here)
similarity = torch.nn.functional.cosine_similarity(question_emb, context_emb)

print(similarity)

# Load the 'squad' dataset
squad_dataset = load_dataset('squad')

# Let's consider a subset for demonstration
squad_dataset = {
    'train': squad_dataset['train'].select(range(1000)),
    'validation': squad_dataset['validation'].select(range(500)),
}

# Load the DPR tokenizers and models
question_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

question_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
ctx_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

def prepare_dataset(example):
    # Tokenize questions and contexts
    question = question_tokenizer(example['question'], truncation=True, padding='max_length', max_length=256)
    context = ctx_tokenizer(example['context'], truncation=True, padding='max_length', max_length=256)

    return {
        'question_input_ids': torch.tensor(question['input_ids']),
        'question_attention_mask': torch.tensor(question['attention_mask']),
        'context_input_ids': torch.tensor(context['input_ids']),
        'context_attention_mask': torch.tensor(context['attention_mask']),
        'labels': torch.tensor(0),  # for SQuAD, we don't have labels as in NQ, so just a dummy label
    }

# Prepare datasets
squad_dataset = {
    'train': squad_dataset['train'].map(prepare_dataset),
    'validation': squad_dataset['validation'].map(prepare_dataset),
}

# Convert to PyTorch DataLoader
dataloader = {
    'train': DataLoader(squad_dataset['train'], batch_size=16, shuffle=True),
    'validation': DataLoader(squad_dataset['validation'], batch_size=16, shuffle=True),
}

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
question_model = question_model.to(device)
ctx_model = ctx_model.to(device)

# Initialize optimizer
optimizer = AdamW(list(question_model.parameters()) + list(ctx_model.parameters()), lr=1e-5)

# Training loop
for epoch in range(3):
    for i, batch in enumerate(dataloader['train']):
        # Move the batch tensors to the same device as the models
        question_input_ids = torch.stack(batch['question_input_ids']).to(device)
        question_attention_mask = torch.stack(batch['question_attention_mask']).to(device)
        context_input_ids = torch.stack(batch['context_input_ids']).to(device)
        context_attention_mask = torch.stack(batch['context_attention_mask']).to(device)

        optimizer.zero_grad()

        # forward pass
        question_outputs = question_model(input_ids=question_input_ids, attention_mask=question_attention_mask)
        ctx_outputs = ctx_model(input_ids=context_input_ids, attention_mask=context_attention_mask)

        # Calculate the loss
        loss = ((question_outputs.pooler_output - ctx_outputs.pooler_output)**2).mean()

        print(f"Batch {i}, Loss: {loss.item()}")

        # backward pass
        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch} completed.")