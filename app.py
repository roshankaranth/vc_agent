import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/chat"

st.title("VC Agent Chat Interface")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    payload = {
        "message": user_input,
        "session_id": st.session_state.session_id
    }

    try:
        res = requests.post(API_URL, json=payload)
        res.raise_for_status()
        data = res.json()
        assistant_reply = data.get("response", "No reply returned")

        st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    except Exception as e:
        st.error(f"Backend error: {e}")
        st.session_state.history.append({
            "role": "assistant",
            "content": "Error communicating with backend."
        })

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
