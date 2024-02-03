from transformers import DistilBertTokenizer, DistilBertModel
from scipy.spatial.distance import cosine
import torch

def calculate_sentence_similarity(sentence1, sentence2):
    # Load DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Tokenize input sentences
    tokens = tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors='pt')

    # Get sentence embeddings
    with torch.no_grad():
        outputs = model(**tokens)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    # Calculate cosine similarity
    similarity = 1 - cosine(sentence_embeddings[0], sentence_embeddings[1])
    return similarity

sentence1 = "I like cats"
sentence2 = "There are some boys playing football"

similarity_score = calculate_sentence_similarity(sentence1, sentence2)
print("Similarity score:", similarity_score)