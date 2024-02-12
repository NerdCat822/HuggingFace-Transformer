from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "I am feeling happy and sad at the same time."
labels = ["normal", "happy", "sad"]

result = classifier(text, labels)

model_name = "openai/clip-vit-large-patch14-336"
classifier = pipeline("zero-shot-image-classification", model = model_name)

image_to_classify = "/content/istockphoto-877369086-1024x1024.jpg"
labels_for_classification =  ["zebra",
                              "lion",
                              "dog",
                              "cat"]
scores = classifier(image_to_classify,
                    candidate_labels = labels_for_classification)