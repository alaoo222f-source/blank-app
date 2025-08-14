""" Simple Chat AI (Streamlit + OpenRouter)

Quickstart

1. pip install streamlit requests


2. Set your key:

export OPENROUTER_API_KEY=sk-... or create .streamlit/secrets.toml with: OPENROUTER_API_KEY = "sk-..."



3. Run: streamlit run streamlit_app.py



Notes

Model defaults to openrouter/auto, but you can pick others.

We store chat history in session_state. No server-side storage. """


from future import annotations import os import json import requests import streamlit as st

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

---------- Helpers ----------

def get_api_key() -> str | None: # Prefer Streamlit secrets, fallback to env var key = st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") else None if not key: key = os.getenv("OPENROUTER_API_KEY") return key

def call_openrouter(api_key: str, model: str, messages: list[dict], temperature: float = 0.3, max_tokens: int = 1000) -> str: headers = { "Authorization": f"Bearer {api_key}", "HTTP-Referer": "https://simple-openrouter-chat.local/",  # set your deployed URL if any "X-Title": "Simple OpenRouter Chat", "Content-Type": "application/json", } payload = { "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, } resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=60) if resp.status_code != 200: # surface helpful error info return f"❌ OpenRouter error {resp.status_code}: {resp.text[:500]}" data = resp.json() try: return data["choices"][0]["message"]["content"] except Exception: return json.dumps(data, ensure_ascii=False, indent=2)

---------- UI ----------

st.set_page_config(page_title="Simple OpenRouter Chat", page_icon="💬", layout="centered") st.title("💬 Simple OpenRouter Chat")

with st.sidebar: st.header("Settings") api_key = st.text_input("OpenRouter API Key", value=get_api_key() or "", type="password") model = st.selectbox( "Model", [ "openrouter/auto", "anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini", "google/gemini-1.5-pro", "meta-llama/llama-3.1-70b-instruct", ], index=0, ) temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05) sys_default = "You are a helpful, concise assistant. Reply in the user's language." system_prompt = st.text_area("System Prompt", value=sys_default, height=80)

init chat state

if "chat" not in st.session_state: st.session_state.chat = []  # list of {role, content}

render history

for m in st.session_state.chat: with st.chat_message(m["role"]): st.markdown(m["content"])  # supports markdown

input box

user_msg = st.chat_input("Type your message…")

if user_msg: if not api_key: st.error("Please provide your OpenRouter API key in the sidebar.") else: # append user st.session_state.chat.append({"role": "user", "content": user_msg})

# build message list for API (system + history)
    messages = ([{"role": "system", "content": system_prompt}] + st.session_state.chat)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = call_openrouter(api_key, model, messages, temperature=temperature)
            st.session_state.chat.append({"role": "assistant", "content": reply})
            st.markdown(reply)

small toolbar

col1, col2 = st.columns(2) with col1: if st.button("🧹 Clear Chat"): st.session_state.chat = [] st.experimental_rerun() with col2: if st.button("📋 Copy Last Reply") and st.session_state.chat: # just show a code block to make copy easier last = st.session_state.chat[-1]["content"] if st.session_state.chat[-1]["role"] == "assistant" else "" st.code(last or "(no assistant message yet)")

st.caption("Tip: set OPENROUTER_API_KEY in your environment or Streamlit secrets for convenience.")

