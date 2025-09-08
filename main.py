import os
import json
import discord
from discord.ext import commands
from keep_alive import keep_alive  # Optional Flask server
from sentence_transformers import SentenceTransformer
import faiss

# ---- Keep bot alive via Flask server (optional) ----
keep_alive()

# ---- Load environment variables ----
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# ---- Bot setup ----
intents = discord.Intents.default()
intents.messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ---- Load FAISS memory ----
memory_file = "memory.json"
if not os.path.exists(memory_file):
    with open(memory_file, "w") as f:
        json.dump([], f)

with open(memory_file, "r") as f:
    memory_data = json.load(f)

# Sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Build FAISS index
if memory_data:
    embeddings = [model.encode(entry['text']) for entry in memory_data]
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    for vec in embeddings:
        index.add(vec.reshape(1, -1))
else:
    index = None

# ---- Commands ----
@bot.event
async def on_ready():
    print(f"✅ Guild Bot online as {bot.user}")

@bot.command()
async def addknowledge(ctx, *, text):
    global memory_data, index
    new_id = len(memory_data)
    memory_data.append({"id": new_id, "text": text})
    with open(memory_file, "w") as f:
        json.dump(memory_data, f, indent=4)
    # Add embedding to FAISS
    if index is None:
        dim = len(model.encode(text))
        index = faiss.IndexFlatL2(dim)
    index.add(model.encode(text).reshape(1, -1))
    await ctx.send(f"Knowledge added with ID {new_id}.")

@bot.command()
async def editknowledge(ctx, id: int, *, new_text):
    global memory_data, index
    for entry in memory_data:
        if entry['id'] == id:
            entry['text'] = new_text
            break
    with open(memory_file, "w") as f:
        json.dump(memory_data, f, indent=4)
    # Rebuild FAISS index
    embeddings = [model.encode(entry['text']) for entry in memory_data]
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    for vec in embeddings:
        index.add(vec.reshape(1, -1))
    await ctx.send(f"Knowledge ID {id} updated.")

@bot.command()
async def query(ctx, *, question):
    if index is None:
        await ctx.send("No knowledge stored yet.")
        return
    q_vec = model.encode(question).reshape(1, -1)
    D, I = index.search(q_vec, k=3)  # Return top 3 results
    responses = [memory_data[i]['text'] for i in I[0]]
    await ctx.send("\n\n".join(responses))

# ---- Run Bot ----
bot.run(DISCORD_TOKEN)