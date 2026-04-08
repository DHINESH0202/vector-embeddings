from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dataset 
texts = [
    "Online shopping is very convenient",
    "E-commerce platforms offer many products",
    "I enjoy watching movies at night",
    "Films are a great source of entertainment",
    "Healthy food improves our lifestyle",
    "Eating vegetables is good for health"
]

# Convert to embeddings
embeddings = model.encode(texts).astype('float32')

# Normalize (for cosine similarity)
faiss.normalize_L2(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# Add embeddings
index.add(embeddings)

print("✅ Vector database created successfully!\n")

# Query (fixed to avoid input issue)
query = "online shopping products"

# Convert query
query_vec = model.encode([query]).astype('float32')
faiss.normalize_L2(query_vec)

# Search
k = 3
scores, indices = index.search(query_vec, k)

# Print + Save output
print("===== SEARCH RESULTS =====\n")

with open("search_results2.txt", "w") as f:
    f.write("===== VECTOR SEARCH RESULTS =====\n\n")
    f.write(f"Query: {query}\n\n")

    for rank, i in enumerate(indices[0]):
        sentence = texts[i]
        score = scores[0][rank]

        output = f"{rank+1}. {sentence}\n   Similarity Score: {score:.4f}\n\n"
        
        print(output)
        f.write(output)

print("✅ Results saved to search_results2.txt")