# server.py
from __future__ import annotations
import os
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import ReasoningAgent, AgentReply

# ---- Pydantic models (JSON-safe) ----
class ToolEventModel(BaseModel):
    name: str
    input: str
    ok: bool
    output: str

class AgentReplyModel(BaseModel):
    text: str
    tool_events: List[ToolEventModel]

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []

# ---- App ----
app = FastAPI(title="AgentKit Backend", version="1.0.0")

# Allow Streamlit on localhost to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo; lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create one agent per process (stateless API: history comes from client)
AGENT = ReasoningAgent()

@app.post("/chat", response_model=AgentReplyModel)
def chat(req: ChatRequest) -> AgentReplyModel:
    reply: AgentReply = AGENT.chat(req.message, history=req.history or [])
    # Convert dataclass ToolEvent -> Pydantic model
    events = [
        ToolEventModel(
            name=e.name, input=e.input, ok=e.ok, output=e.output
        ) for e in (reply.tool_events or [])
    ]
    return AgentReplyModel(text=reply.text, tool_events=events)

@app.get("/health")
def health():
    return {"ok": True}
