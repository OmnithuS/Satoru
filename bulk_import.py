import json
from knowledge_base import embedder, index, knowledge_texts

# --- Load memory ---
with open("memory.json", "r") as f:
    memory = json.load(f)

# --- Load bulk facts from a text file ---
# Each line in bulk_knowledge.txt = one fact
with open("bulk_knowledge.txt", "r") as f:
    new_facts = [line.strip() for line in f if line.strip()]

# --- Add to memory ---
memory["knowledge_base"].extend(new_facts)

# --- Save updated memory ---
with open("memory.json", "w") as f:
    json.dump(memory, f, indent=4)

# --- Rebuild FAISS index ---
knowledge_texts.extend(new_facts)  # add to in-memory list
embeddings = embedder.encode(knowledge_texts, convert_to_numpy=True)
index.reset()  # clear old index
index.add(embeddings)

print(f"✅ Added {len(new_facts)} new knowledge items and rebuilt FAISS index.")
