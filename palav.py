# -*- coding: utf-8 -*-
import os
import time
import streamlit as st
from openai import OpenAI

import streamlit.components.v1 as components

#  NEW: periodic reruns so we can detect inactivity
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False


def scroll_to_bottom():
    components.html(
        """
        <script>
          const scroll = () => window.scrollTo(0, document.body.scrollHeight);
          scroll();
          setTimeout(scroll, 50);
          setTimeout(scroll, 150);
          setTimeout(scroll, 300);
        </script>
        """,
        height=0,
    )

st.set_page_config(page_title="Breastfeeding Manual Chatbot", layout="centered")
# removing the side bar from UI
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Palav Breastfeeding Userguide ")

# --- CONFIG ---
API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_VS_ID = st.secrets.get("VECTOR_STORE_ID", os.getenv("VECTOR_STORE_ID", ""))

if not API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to Streamlit secrets or your environment variables.")
    st.stop()

client = OpenAI(api_key=API_KEY)

with st.sidebar:
    st.header("Settings")
    vector_store_id = st.text_input("Vector Store ID", value=DEFAULT_VS_ID, placeholder="vs_...")
    model = st.selectbox("Model", ["gpt-4.1", "gpt-4.1-mini"], index=0)
    show_citations = st.checkbox("Show citations/debug", value=False)

    st.divider()
    st.markdown(
        "**Tip:** Put these in `.streamlit/secrets.toml`:\n\n"
        "- `OPENAI_API_KEY`\n"
        "- `VECTOR_STORE_ID`"
    )

if not vector_store_id:
    st.warning("Enter your Vector Store ID in the sidebar (it starts with `vs_`).")
    st.stop()

SYSTEM_INSTRUCTIONS = (
    "You are an NGO breastfeeding education assistant. "
    "Answer ONLY using the uploaded manual from file_search. "
    "If the manual does not cover the topic, say: 'The manual does not cover this. I can only help with breastfeeding related topic/question' "
    "Use clear, parent-friendly language. "
    "Include what to do / what not to do when relevant. "
)

# --- NEW: Idle timeout config ---
IDLE_SECONDS = 120                 # 2 minutes
REFRESH_INTERVAL_MS = 10_000       # rerun every 10 seconds

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to Palav Breastfeeding Guide Chatbot\n\n"
                "I am here to help you find clear, reliable information from Palav breastfeeding user guide.\n\n"
                "You can ask questions or type a topic such as:\n"
                "- early milk or colostrum\n"
                "- Attachment and Latch\n"
                "- Hand Expression Techniques\n"
                "- Feeding frequency and hunger cues\n"
                "- Do and dont for breastfeeding mothers\n"
                "- When to seek medical help\n\n"
                "You can also type any breastfeeding-related topic you would like to learn more about."
            )
        }
    ]

# NEW: track last user activity + prevent repeated idle messages
if "last_user_activity_ts" not in st.session_state:
    st.session_state.last_user_activity_ts = time.time()

if "idle_nudge_sent" not in st.session_state:
    st.session_state.idle_nudge_sent = False

#  NEW: trigger periodic reruns so we can detect idle time passing
if HAS_AUTOREFRESH:
    st_autorefresh(interval=REFRESH_INTERVAL_MS, key="idle_autorefresh")
else:
    # Fallback (no extra package): refresh page using HTML meta refresh.
    # Less ideal, but works if you don't want to install streamlit-autorefresh.
    components.html(
        f"<meta http-equiv='refresh' content='{REFRESH_INTERVAL_MS/1000}'>",
        height=0,
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Type your question")

def extract_citations(resp) -> list[str]:
    citations = []
    try:
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                for ann in getattr(c, "annotations", []) or []:
                    parts = []
                    for key in ("filename", "file_id", "quote", "text"):
                        val = getattr(ann, key, None) or (ann.get(key) if isinstance(ann, dict) else None)
                        if val:
                            parts.append(f"{key}={val}")
                    citations.append(", ".join(parts) if parts else str(ann))
    except Exception:
        pass
    return citations

#  NEW: Idle check (only when user did NOT just submit something)
if not prompt:
    idle_for = time.time() - st.session_state.last_user_activity_ts
    if idle_for >= IDLE_SECONDS and not st.session_state.idle_nudge_sent:
        nudge = "Just checking in . do you have any more questions or would you like to know more about any breastfeeding topic?"
        st.session_state.messages.append({"role": "assistant", "content": nudge})
        st.session_state.idle_nudge_sent = True
        st.experimental_rerun()

if prompt:
    #  user interacted: update last activity + allow future nudges again
    st.session_state.last_user_activity_ts = time.time()
    st.session_state.idle_nudge_sent = False

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the manual"):
            input_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

            resp = client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=input_messages,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                }],
            )

            answer = getattr(resp, "output_text", "").strip() or "(No output received.)"
            st.markdown(answer)

            if show_citations:
                cites = extract_citations(resp)
                with st.expander("Citations / Debug"):
                    if cites:
                        for i, c in enumerate(cites, 1):
                            st.write(f"{i}. {c}")
                    else:
                        st.write("No citations found in this response object.")

    # Save assistant answer
    st.session_state.messages.append({"role": "assistant", "content": answer})

scroll_to_bottom()
