"""
vornis_chatbot.py - AI Chatbot module for VornisAI (Streamlit + OpenAI).

The chatbot can call three tools (defined in tools.py):
    get_planet_data, compare_planets, predict_mission_feasibility

Usage:
    from vornis_chatbot import render_vornis_chatbot
    render_vornis_chatbot()

Env vars:
    OPENAI_API_KEY   (required)
    CHATBOX_MODEL    (required) model name, see https://platform.openai.com/docs/models
    OPENAI_BASE_URL  (optional) point at an OpenAI-compatible provider
"""

import json
import os

import openai
import streamlit as st

from tools import TOOLS, TOOL_FUNCTIONS

MAX_TOOL_ROUNDS = 5
MAX_HISTORY = 30

SYSTEM_PROMPT = """You are Vornis AI, the assistant inside VornisAI, a platform for \
planetary analysis, planet comparison and ML-based mission feasibility prediction.

- Use your tools whenever the user asks about specific planets, comparisons, or \
whether a mission is feasible. Do not guess numbers that a tool can provide.
- Explain results in clear language, state units, and mention that feasibility \
scores are model estimates, not guarantees.
- If a tool returns an error or the planet is unknown, say so plainly.
- For general space-science questions you may answer directly. Keep answers concise."""

# tools.py uses a neutral schema; convert it to OpenAI's function format here
# so teammates never have to touch provider-specific code.
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]


@st.cache_resource
def _client() -> openai.OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("OPENAI_API_KEY is not set. Add it to .env (local) or Railway > Variables.")
        st.stop()
    return openai.OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None)


def _model() -> str:
    model = os.getenv("CHATBOX_MODEL")
    if not model:
        st.error("CHATBOX_MODEL is not set. Pick a model from platform.openai.com/docs/models.")
        st.stop()
    return model


def _run_tool(name: str, raw_args: str) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool {name}"})
    try:
        return json.dumps(fn(**json.loads(raw_args or "{}")))
    except Exception as e:  # never let a tool crash the chat
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def _trim(history: list) -> list:
    """Keep the last N messages, never starting on an orphaned tool result."""
    h = history[-MAX_HISTORY:]
    while h and h[0]["role"] != "user":
        h = h[1:]
    return h


def render_vornis_chatbot(key: str = "vornis_chat") -> None:
    ui_key, api_key = f"{key}_ui", f"{key}_api"
    st.session_state.setdefault(ui_key, [])   # what the user sees
    st.session_state.setdefault(api_key, [])  # full API history incl. tool messages

    top = st.columns([5, 1])
    top[0].subheader("Vornis AI Assistant")
    if top[1].button("Clear", key=f"{key}_clear"):
        st.session_state[ui_key], st.session_state[api_key] = [], []
        st.rerun()

    if not st.session_state[ui_key]:
        st.caption("Try: 'Compare Mars and Venus' or 'Is a Mars lander feasible?'")

    for m in st.session_state[ui_key]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask about planets, comparisons, or mission feasibility...", key=f"{key}_input")
    if not prompt:
        return

    model = _model()
    st.session_state[ui_key].append({"role": "user", "content": prompt})
    st.session_state[api_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        shown = ""
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                stream = _client().chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                    + _trim(st.session_state[api_key]),
                    tools=OPENAI_TOOLS,
                    stream=True,
                )

                text, calls = "", {}
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        text += delta.content
                        shown += delta.content
                        placeholder.markdown(shown + "▌")
                    for tc in delta.tool_calls or []:
                        c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            c["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                c["name"] = tc.function.name
                            if tc.function.arguments:
                                c["args"] += tc.function.arguments

                assistant_msg = {"role": "assistant", "content": text or None}
                if calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"] or "{}"},
                        }
                        for _, c in sorted(calls.items())
                    ]
                st.session_state[api_key].append(assistant_msg)

                if not calls:
                    break

                for _, c in sorted(calls.items()):
                    with st.status(f"Running {c['name']}...", expanded=False) as status:
                        out = _run_tool(c["name"], c["args"])
                        st.code(out, language="json")
                        status.update(label=f"Used {c['name']}", state="complete")
                    st.session_state[api_key].append(
                        {"role": "tool", "tool_call_id": c["id"], "content": out}
                    )
                shown += "\n\n" if shown else ""

            placeholder.markdown(shown)
        except openai.AuthenticationError:
            shown = "The API key was rejected. Check OPENAI_API_KEY."
            placeholder.error(shown)
        except openai.NotFoundError:
            shown = "That model name isn't available to this key. Check CHATBOX_MODEL."
            placeholder.error(shown)
        except openai.RateLimitError:
            shown = "Too many requests, or the account is out of credit. Try again shortly."
            placeholder.warning(shown)
        except openai.APIError as e:
            shown = "Sorry, I couldn't reach the AI service."
            placeholder.error(f"{shown} ({type(e).__name__})")

    st.session_state[ui_key].append({"role": "assistant", "content": shown})


if __name__ == "__main__":
    st.set_page_config(page_title="Vornis AI", page_icon="🪐")
    render_vornis_chatbot()
