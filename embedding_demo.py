from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences
sentences = [
    "I love artificial intelligence",
    "I like AI",
    "The weather is very hot today",
    "It is a sunny day"
]

# Convert to embeddings
embeddings = model.encode(sentences)

# Save to text file
with open("embeddings_output.txt", "w") as f:
    f.write("===== SENTENCE EMBEDDINGS =====\n\n")
    
    for i in range(len(sentences)):
        f.write(f"Sentence {i+1}: {sentences[i]}\n")
        f.write(f"Embedding (first 5 values): {embeddings[i][:5]}\n")
        f.write("-" * 50 + "\n")

print("✅ Output saved to embeddings_output.txt")