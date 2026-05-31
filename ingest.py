import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from tqdm import tqdm

# --- FOOLPROOF ENVIRONMENT LOADING ---
# This forces Python to look in the exact directory where ingest.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
pinecone_key = os.getenv("PINECONE_API_KEY")

# Troubleshooting print statements to tell you EXACTLY what is going on:
print("--- Debugging Key Loader ---")
print(f"Looking for .env at: {env_path}")
print(f"File actually exists there? {os.path.exists(env_path)}")
print(f"OPENAI_API_KEY loaded: {'YES (Starts with ' + api_key[:6] + '...)' if api_key else '❌ NO - IS EMPTY'}")
print(f"PINECONE_API_KEY loaded: {'YES' if pinecone_key else '❌ NO - IS EMPTY'}")
print("----------------------------\n")

if not api_key:
    raise ValueError("❌ Hard Stop: OPENAI_API_KEY is completely missing from memory! Fix your .env file.")

# 1. INITIALIZE PINECONE & EMBEDDING MODEL
embeddings = OpenAIEmbeddings(
    model="4UHRUIN-text-embedding-3-small",
    openai_api_key=api_key,
    openai_api_base=api_base
)

pc = Pinecone(api_key=pinecone_key)
index = pc.Index("medium-articles") # Make sure this matches your exact Pinecone index name
# --- THE REST OF YOUR CHUNKING & LOOPING CODE REMAINS EXACTLY THE SAME ---

# 2. CONFIGURABLE HYPERPARAMETERS
# We use simple word-counting. 1 word ~ 1.3 tokens. 
# 512 words is roughly 650 tokens (safely below the 1024 token limit)
CHUNK_SIZE_WORDS = 512  
OVERLAP_WORDS = 100     

def chunk_text_by_words(text, size, overlap):
    words = str(text).split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        i += (size - overlap)
    return chunks

# 3. LOAD DATA (Budget Check: Load only 50 rows for testing!)
print("Loading dataset...")
SAMPLE_SIZE = 50 
df = pd.read_csv("medium-english-50mb.csv").dropna(subset=['text']).head(SAMPLE_SIZE)

print(f"Processing and uploading chunks for {SAMPLE_SIZE} articles...")
vectors_to_upsert = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    article_id = str(idx)
    article_chunks = chunk_text_by_words(row['text'], CHUNK_SIZE_WORDS, OVERLAP_WORDS)
    
    for chunk_idx, chunk in enumerate(article_chunks):
        # Unique ID combining article index and chunk location
        unique_id = f"{article_id}#_#{chunk_idx}"
        
        # Metadata must accurately reflect what your /api/prompt endpoint will return
        metadata = {
            "article_id": article_id,
            "title": str(row['title']),
            "chunk": chunk,
            "url": str(row['url']),
            "authors": str(row['authors']),
            "tags": str(row['tags'])
        }
        
        try:
            # Generate the 1536-dimension embedding vector
            vector = embeddings.embed_query(chunk)
            vectors_to_upsert.append((unique_id, vector, metadata))
        except Exception as e:
            print(f"\nError creating vector for article {article_id}: {e}")
            
        # Batch uploads to Pinecone in packages of 50 to optimize network stability
        if len(vectors_to_upsert) >= 50:
            index.upsert(vectors=vectors_to_upsert)
            vectors_to_upsert = []

# Upload remaining vectors
if vectors_to_upsert:
    index.upsert(vectors=vectors_to_upsert)

print("\nIngestion complete! Go check your Pinecone web dashboard to see your vectors.")