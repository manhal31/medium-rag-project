import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

# Locate templates directory dynamically for secure Vercel resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# 1. RAG Hyperparameter Config Mapping
RAG_CONFIG = {
    "chunk_size": 512,       
    "overlap_ratio": 0.25,   
    "top_k": 3               
}

# 2. Initialize APIs with the official Technion course proxy models
embeddings = OpenAIEmbeddings(
    model="4UHRUIN-text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

llm = ChatOpenAI(
    model="4UHRUIN-gpt-5-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE")
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("medium-articles")

# 3. Your Requested System Prompt Formulation Constraints
SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and article passages). "
    "You must not use any external knowledge, the open internet, or information that is "
    "not explicitly contained in the retrieved context. If the answer cannot be determined "
    "from the provided context, respond: \"I don't know based on the provided Medium articles data.\" "
    "Always explain your answer using the given context, quoting or paraphrasing the relevant "
    "article passage or metadata when helpful."
)

class QueryPayload(BaseModel):
    question: str

# ENDPOINT 1: POST /api/prompt (Processes questions using RAG context)
@app.post("/api/prompt")
async def execute_rag(payload: QueryPayload):
    query_vector = embeddings.embed_query(payload.question)
    raw_matches = index.query(vector=query_vector, top_k=RAG_CONFIG["top_k"], include_metadata=True)
    
    formatted_context_list = []
    context_text_blocks = []
    
    for match in raw_matches.get("matches", []):
        meta = match.get("metadata", {})
        formatted_context_list.append({
            "article_id": meta.get("article_id"),
            "title": meta.get("title"),
            "chunk": meta.get("chunk"),
            "score": match.get("score")
        })
        context_text_blocks.append(f"Title: {meta.get('title')}\nPassage: {meta.get('chunk')}")
        
    injected_context = "\n\n---\n\n".join(context_text_blocks)
    user_prompt_content = f"Context:\n{injected_context}\n\nQuestion: {payload.question}"
    
    llm_output = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt_content)
    ])
    
    return {
        "response": llm_output.content,
        "context": formatted_context_list,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt_content
        }
    }

# ENDPOINT 2: GET /api/stats (Returns RAG operational settings parameters)
@app.get("/api/stats")
async def deliver_stats():
    return RAG_CONFIG

# ENDPOINT 3: GET / (Serves the clean HTML file cleanly using Jinja2)
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})