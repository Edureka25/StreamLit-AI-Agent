from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from tools import ToolResult, calculator, clock, FactsStore, ToolEvent  # same folder imports

load_dotenv()

@dataclass
class AgentReply:
    text: str
    tool_events: List[ToolEvent]

def _is_greeting(text: str) -> bool:
    return re.match(r"^\s*(hi|hello|hey|hola|namaste|good\s*(morning|afternoon|evening)?)\b", text, re.I) is not None

def _is_followup(text: str) -> bool:
    return bool(re.search(r"\b(explain (this|that)|in brief|briefly|why|how so)\b", text, re.I))

def _last_assistant_text(history: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(history or []):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return None

def _last_user_text(history: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(history or []):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg["content"])
    return None

def _local_brief_explanation(prev_answer: str, prev_user: Optional[str]) -> str:
    if not prev_answer:
        return "I can explain — could you repeat the part you want clarified?"
    if len(prev_answer) <= 160 and prev_answer.count(".") <= 1:
        return f"In short: {prev_answer}"
    sent = prev_answer.split(".")[0].strip()
    if not sent:
        return "Briefly: it follows from the previous result."
    return f"Briefly: {sent}." if not prev_user else f"Briefly: {sent}. (Related to your earlier message: “{prev_user}”.)"

class ReasoningAgent:
    """
    - Natural greetings
    - Tool routing (math/time/memory)
    - Conversational memory for follow-ups using recent history
    - OpenAI small-talk/clarification if OPENAI_API_KEY is set
    """
    def __init__(self) -> None:
        self.facts = FactsStore()
        self._client: Optional[OpenAI] = None
        if os.getenv("OPENAI_API_KEY", "").strip():
            self._client = OpenAI()

    def _openai_reply(self, user: str, history: List[Dict[str, Any]]) -> Optional[str]:
        if not self._client:
            return None
        try:
            msgs: List[Dict[str, str]] = [{
                "role": "system",
                "content": "You are a warm, concise assistant inside a teaching demo. "
                           "Use chat history for context. Be brief (1–3 sentences). Avoid chain-of-thought."
            }]
            compact = [{"role": m["role"], "content": m["content"]}
                       for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
            msgs.extend(compact[-6:])
            msgs.append({"role": "user", "content": user})
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msgs,
                temperature=0.6,
                max_tokens=160,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return None

    def chat(self, user: str, history: Optional[List[Dict[str, Any]]] = None) -> AgentReply:
        events: List[ToolEvent] = []
        history = history or []

        # 0) Greeting
        if _is_greeting(user):
            text = self._openai_reply(user, history) or \
                   "Hello! I remember this session. Ask me anything, or say “help” to see capabilities."
            return AgentReply(text=text, tool_events=events)

        # 1) Memory: remember key = value
        m = re.match(r"^\s*(remember|save)\s+(.+?)\s*=\s*(.+?)\s*$", user, flags=re.I)
        if m:
            _, key, val = m.groups()
            r: ToolResult = self.facts.remember(key, val)
            events.append(r.event)
            return AgentReply(text="Got it — I saved that.", tool_events=events)

        # 2) Memory: recall key
        m = re.match(r"^\s*(recall|what did i save for)\s+(.+?)\s*$", user, flags=re.I)
        if m:
            _, key = m.groups()
            r = self.facts.recall(key)
            events.append(r.event)
            return AgentReply(text=r.content if r.ok else "I don't have that saved yet.", tool_events=events)

        # 3) Time
        if re.search(r"\b(time|date|now)\b", user, flags=re.I):
            r = clock()
            events.append(r.event)
            return AgentReply(text=f"The time is {r.content}.", tool_events=events)

        # 4) Calculator
        calc_match = re.match(r"^\s*calculate\s+(.+)$", user, flags=re.I)
        expr = calc_match.group(1) if calc_match else None
        if expr is None and re.match(r"^[\d\.\s\+\-\*\/\(\)\%]+$", user):
            expr = user.strip()
        if expr:
            r = calculator(expr)
            events.append(r.event)
            return AgentReply(text=r.content if r.ok else "That expression didn't work for me.", tool_events=events)

        # 5) Follow-up clarification
        if _is_followup(user):
            prev_answer = _last_assistant_text(history) or ""
            prev_user = _last_user_text(history)
            ai = self._openai_reply(f"{user}\n\nContext:\nPrevious answer: {prev_answer}", history)
            if ai:
                return AgentReply(text=ai, tool_events=events)
            return AgentReply(text=_local_brief_explanation(prev_answer, prev_user), tool_events=events)

        # 6) General fallback
        ai = self._openai_reply(user, history)
        if ai:
            return AgentReply(text=ai, tool_events=events)

        return AgentReply(
            text=("I keep our chat in memory for this session. "
                  "I can do quick math (e.g., `12*(3+4)`), tell the time, remember simple facts "
                  "(e.g., `remember project = Apollo` → `recall project`), and handle follow-ups "
                  "like “explain this in brief?”. For richer chat, set OPENAI_API_KEY in a .env file."),
            tool_events=events,
        )
