import discord
import json
import os
from ollama import Ollama
from dotenv import load_dotenv
from keep_alive import keep_alive
from knowledge_base import search_knowledge, embedder, index, knowledge_texts

# Load env vars
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Replace with YOUR Discord user ID
BOT_OWNER_ID = 639200660604321803

# Discord setup
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Ollama model
model = Ollama(model="gpt-oss:20b")

# Memory load/save
def load_memory():
    try:
        with open("memory.json", "r") as f:
            return json.load(f)
    except:
        return {"users": {}, "knowledge_base": []}

def save_memory(memory):
    with open("memory.json", "w") as f:
        json.dump(memory, f, indent=4)

memory = load_memory()

# Keep alive
keep_alive()

@client.event
async def on_ready():
    print(f"✅ Guild Bot online as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_id = str(message.author.id)

    # --- Guild Master Commands ---
    if message.author.id == BOT_OWNER_ID:

        # Add knowledge
        if message.content.startswith("!addknowledge"):
            new_fact = message.content[len("!addknowledge "):].strip()
            if new_fact:
                memory["knowledge_base"].append(new_fact)
                save_memory(memory)

                new_vec = embedder.encode([new_fact], convert_to_numpy=True)
                index.add(new_vec)
                knowledge_texts.append(new_fact)

                await message.channel.send(f"📚 Added to knowledge base: `{new_fact}`")
            else:
                await message.channel.send("⚠️ Please provide text after `!addknowledge`.")
            return

        # List knowledge
        if message.content.startswith("!listknowledge"):
            kb = memory["knowledge_base"]
            if not kb:
                await message.channel.send("📖 Knowledge base is empty.")
            else:
                response = "\n".join([f"{i}. {fact}" for i, fact in enumerate(kb[:10])])
                await message.channel.send(f"📚 Knowledge Base (first 10):\n{response}")
            return

        # Remove knowledge
        if message.content.startswith("!removeknowledge"):
            try:
                index_to_remove = int(message.content.split(" ")[1])
                removed_fact = memory["knowledge_base"].pop(index_to_remove)
                save_memory(memory)

                # Rebuild index
                knowledge_texts.pop(index_to_remove)
                embeddings = embedder.encode(knowledge_texts, convert_to_numpy=True)
                index.reset()
                index.add(embeddings)

                await message.channel.send(f"🗑️ Removed: `{removed_fact}`")
            except:
                await message.channel.send("⚠️ Usage: `!removeknowledge <index>`")
            return

        # Edit knowledge
        if message.content.startswith("!editknowledge"):
            try:
                parts = message.content.split(" ", 2)
                index_to_edit = int(parts[1])
                new_fact = parts[2].strip()

                old_fact = memory["knowledge_base"][index_to_edit]
                memory["knowledge_base"][index_to_edit] = new_fact
                save_memory(memory)

                knowledge_texts[index_to_edit] = new_fact
                embeddings = embedder.encode(knowledge_texts, convert_to_numpy=True)
                index.reset()
                index.add(embeddings)

                await message.channel.send(f"✏️ Updated knowledge:\n`{old_fact}` → `{new_fact}`")
            except:
                await message.channel.send("⚠️ Usage: `!editknowledge <index> <new text>`")
            return

    # --- Regular Conversation ---
    if client.user.mentioned_in(message):
        if user_id not in memory["users"]:
            memory["users"][user_id] = {"messages": [], "rank": "E-Rank 🌱✨"}

        memory["users"][user_id]["messages"].append(message.content)
        save_memory(memory)

        user_history = "\n".join(memory["users"][user_id]["messages"][-5:])
        knowledge = search_knowledge(message.content, top_k=5)

        prompt = f"""
You are the Guild Instructor, a wise and patient mentor.
User rank: {memory['users'][user_id]['rank']}
Relevant knowledge: {knowledge}
User’s recent history: {user_history}

Respond as a roleplay character, always helpful and instructive.
"""

        reply = model.generate(prompt)
        await message.channel.send(reply)

client.run(DISCORD_TOKEN)