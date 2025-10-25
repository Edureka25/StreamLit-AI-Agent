# ui_app.py
import os
import requests
import streamlit as st
from typing import Dict, Any, List

APP_TITLE = "Reasoning Agent • Frontend ↔ Backend Demo"
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{'role': 'user'|'assistant', 'content': str, 'trace': [...] }]

def sidebar():
    st.sidebar.title("Controls")

    st.sidebar.divider()
    st.sidebar.checkbox("Show tool trace", value=True, key="show_trace")

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

def render_header():
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:16px;background:linear-gradient(90deg,#f7f7ff,#f1f5ff);">
          <h2 style="margin:0;">{APP_TITLE}</h2>
          <p style="margin:0.25rem 0 0;color:#444;">
            Streamlit frontend calling a FastAPI AgentKit backend.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_history():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trace") and st.session_state.get("show_trace"):
                with st.expander("Run trace", expanded=False):
                    for i, ev in enumerate(m["trace"], start=1):
                        st.markdown(
                            f"- **{i}. Tool:** `{ev.get('name','')}`  \n"
                            f"  **Input:** `{ev.get('input','')}`  \n"
                            f"  **OK:** `{ev.get('ok','')}`  \n"
                            f"  **Output:** `{ev.get('output','')}`"
                        )

def call_backend_chat(message: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    url = f"{BACKEND_URL.rstrip('/')}/chat"
    try:
        resp = requests.post(url, json={"message": message, "history": history}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"text": f"Backend error: {e}", "tool_events": []}

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()
    sidebar()
    render_header()
    st.divider()
    render_history()

    user = st.chat_input("Type a message (e.g., 12*(3+4), time, remember x=y, 'explain this in brief')...")
    if user:
        # Show user msg
        st.session_state.messages.append({"role": "user", "content": user})
        with st.chat_message("assistant"):
            # Send full client-side history to backend (stateless API)
            reply = call_backend_chat(user, history=st.session_state.messages)
            text = reply.get("text", "")
            trace = reply.get("tool_events", [])
            st.markdown(text)
            if trace and st.session_state.get("show_trace"):
                with st.expander("Run trace", expanded=False):
                    for i, ev in enumerate(trace, start=1):
                        st.markdown(
                            f"- **{i}. Tool:** `{ev.get('name','')}`  \n"
                            f"  **Input:** `{ev.get('input','')}`  \n"
                            f"  **OK:** `{ev.get('ok','')}`  \n"
                            f"  **Output:** `{ev.get('output','')}`"
                        )
        st.session_state.messages.append({"role": "assistant", "content": text, "trace": trace})
        st.rerun()

if __name__ == "__main__":
    main()
