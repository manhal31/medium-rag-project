import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

RAG_CONFIG = {
    "chunk_size": 513,       
    "overlap_ratio": 0.15,   
    "top_k": 8               
}

SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and article passages). "
    "You must not use any external knowledge, the open internet, or information that is "
    "not explicitly contained in the retrieved context. If the answer cannot be determined "
    "from the provided context, respond: \"I don't know based on the provided Medium articles data.\""
    "Always explain your answer using the given context, quoting or paraphrasing the relevant "
    "article passage or metadata when helpful. "
    
    "\n\n[ADVANCED REASONING & COGNITIVE GUIDELINES]:\n"
    "1. CONCEPTUAL SYNONYMS: Do not rely solely on literal keyword matching. If a user asks about a highly specific historical example "
    "(e.g., 'the bubonic plague' or 'a specific company') that is missing, but the retrieved context contains extensive material on the "
    "overarching concept (e.g., 'pandemics/epidemics' or 'business strategy'), you must pivot gracefully to address the conceptual intent.\n"
    "2. TRANSPARENT BRIDGING: When executing a conceptual pivot, explicitly state what is missing first, and then immediately deliver the "
    "rich data you DO have. (Example: 'While the provided articles do not explicitly mention the bubonic plague, the dataset contains clear "
    "evidence regarding how recent pandemics like the Coronavirus spur systemic innovation and recovery...').\n"
    "3. CONCISE STRUCTURE: Deliver the extracted arguments or summaries cleanly using bullet points. Do not write endless conversational "
    "paragraphs explaining your search process.\n"
    "4. MAX LENGTH: Keep your entire response under 200 words total. Cut out all conversational filler.\n"
    "5. CONCEPTUAL BRIDGING: If a specific historical keyword is missing but the context heavily answers the macro concept, explicitly note the missing term in one brief sentence, then deliver the data using bullet points.\n"
    "6. NO META-YAPPING: Do not explain your search process, do not list what you searched for, and never use phrases like 'Based on the provided excerpts, I found...'. Jump straight to the answer.\n"
)

class QueryPayload(BaseModel):
    question: str

@app.post("/api/prompt")
def execute_rag(payload: QueryPayload):
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

@app.get("/api/stats")
def deliver_stats():
    return RAG_CONFIG

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medium Article Research Assistant</title>
        <style>
            * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; }
            body { background: #f8f9fa; height: 100vh; display: flex; overflow: hidden; color: #1a1a1a; justify-content: center; }
            .main-layout { display: flex; width: 100%; max-width: 900px; height: 100vh; background: #ffffff; }
            .chat-pane { flex: 1; display: flex; flex-direction: column; background: #ffffff; height: 100%; box-shadow: 0 0 20px rgba(0,0,0,0.03); }
            .pane-header { padding: 20px 30px; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; justify-content: space-between; background: #ffffff; }
            .pane-header h2 { font-size: 1.15rem; font-weight: 700; color: #111827; letter-spacing: -0.3px; }
            .status-badge { background: #e6f7f0; color: #02946c; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 9999px; text-transform: uppercase; }
            .chat-messages { flex: 1; padding: 30px; overflow-y: auto; display: flex; flex-direction: column; gap: 24px; background: #ffffff; }
            .msg-row { display: flex; gap: 16px; max-width: 85%; animation: slideUp 0.25s ease-out forwards; }
            @keyframes slideUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
            .msg-row.user { align-self: flex-end; flex-direction: row-reverse; }
            .msg-row.bot { align-self: flex-start; }
            .avatar { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; flex-shrink: 0; }
            .msg-row.user .avatar { background: #2563eb; color: white; }
            .msg-row.bot .avatar { background: #111827; color: #03a87c; border: 1px solid #1f2937; }
            .bubble { padding: 14px 20px; border-radius: 14px; font-size: 0.98rem; line-height: 1.6; white-space: pre-line; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
            .msg-row.user .bubble { background: #2563eb; color: white; border-top-right-radius: 2px; }
            .msg-row.bot .bubble { background: #f3f4f6; color: #1f2937; border-top-left-radius: 2px; border: 1px solid #e5e7eb; }
            .input-bar { padding: 24px 30px; border-top: 1px solid #e5e7eb; background: #ffffff; display: flex; gap: 12px; align-items: center; }
            input { flex: 1; padding: 14px 18px; border: 1px solid #d1d5db; border-radius: 10px; font-size: 0.98rem; outline: none; transition: all 0.15s ease; background: #f9fafb; }
            input:focus { border-color: #03a87c; background: #ffffff; box-shadow: 0 0 0 3px rgba(3, 168, 124, 0.12); }
            button { background: #111827; color: white; border: none; padding: 14px 24px; border-radius: 10px; font-weight: 600; cursor: pointer; transition: all 0.15s ease; font-size: 0.95rem; }
            button:hover { background: #03a87c; }
            button:disabled { background: #d1d5db; cursor: not-allowed; }
            .dots { display: flex; gap: 4px; padding: 6px 0; }
            .dots span { width: 8px; height: 8px; background: #9ca3af; border-radius: 50%; animation: bounce 1.4s infinite both; }
            .dots span:nth-child(2) { animation-delay: 0.2s; }
            .dots span:nth-child(3) { animation-delay: 0.4s; }
            @keyframes bounce { 0%, 80%, 100% { transform: scale(0.3); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }
        </style>
    </head>
    <body>
    <div class="main-layout">
        <div class="chat-pane">
            <div class="pane-header">
                <h2>💬 Smart Medium Assistant</h2>
                <div class="status-badge">System Online</div>
            </div>
            <div class="chat-messages" id="chatFeed">
                <div class="msg-row bot">
                    <div class="avatar">🤖</div>
                    <div class="bubble">Welcome! I am an AI research assistant trained exclusively on the Medium article collection. Ask me any question, and I will find the exact articles containing the answer.</div>
                </div>
            </div>
            <div class="input-bar">
                <input type="text" id="promptInput" placeholder="Type your research question here..." onkeypress="handleKey(event)">
                <button id="execBtn" onclick="runPipeline()">Ask Assistant</button>
            </div>
        </div>
    </div>
    <script>
        async function runPipeline() {
            const input = document.getElementById('promptInput');
            const btn = document.getElementById('execBtn');
            const text = input.value.trim();
            if (!text) return;

            postBubble(text, 'user', '👤');
            input.value = '';
            input.disabled = true;
            btn.disabled = true;
            const loader = postLoader();

            try {
                const res = await fetch('/api/prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: text })
                });
                const data = await res.json();
                loader.remove();
                if (data.response) {
                    postBubble(data.response, 'bot', '🤖');
                } else {
                    postBubble("Sorry, something went wrong with the data connection. Please try again.", 'bot', '🤖');
                }
            } catch (e) {
                loader.remove();
                postBubble("Could not connect to the assistant server. Please check your network connection.", 'bot', '🤖');
            }
            input.disabled = false;
            btn.disabled = false;
            input.focus();
        }
        function postBubble(text, side, char) {
            const feed = document.getElementById('chatFeed');
            const row = document.createElement('div');
            row.className = "msg-row " + side;
            row.innerHTML = '<div class="avatar">' + char + '</div><div class="bubble">' + text + '</div>';
            feed.appendChild(row);
            feed.scrollTop = feed.scrollHeight;
        }
        function postLoader() {
            const feed = document.getElementById('chatFeed');
            const row = document.createElement('div');
            row.className = 'msg-row bot';
            row.innerHTML = '<div class="avatar">🤖</div><div class="bubble"><div class="dots"><span></span><span></span><span></span></div></div>';
            feed.appendChild(row);
            feed.scrollTop = feed.scrollHeight;
            return row;
        }
        function handleKey(e) { if (e.key === 'Enter') runPipeline(); }
    </script>
    </body>
    </html>
    """