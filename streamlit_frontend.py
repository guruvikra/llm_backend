import streamlit as st
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from graph import compiled_graph, get_config

st.title("Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "config" not in st.session_state:
    st.session_state.config = get_config()

# Define a callback function to handle form submission
def submit():
    user_input = st.session_state.user_input.strip()
    if user_input:
        # Add human message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Process with LangGraph
        async def stream_response():
            events = compiled_graph.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=st.session_state.config,
                stream_mode="messages"
            )
            response_chunks = []
            async for event in events:
                if isinstance(event, tuple):
                    msg = event[0]
                    if isinstance(msg, AIMessage) and msg.content.strip():
                        response_chunks.append(msg.content)
            return response_chunks

        ai_chunks = asyncio.run(stream_response())
        for chunk in ai_chunks:
            ai_msg = AIMessage(content=chunk)
            st.session_state.chat_history.append(ai_msg)

    # Clear the input field by resetting the session state
    st.session_state.user_input = ""

# Input form with on_change callback
with st.form("chat_form"):
    st.text_input("You:", key="user_input")
    st.form_submit_button("Send", on_click=submit)

# Display all messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You**: {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Assistant**: {msg.content}")
