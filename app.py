import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from dataIngestion import get_rag_chain

load_dotenv()

st.set_page_config(page_title="Galgotias University Bot", page_icon="🎓")
st.title("🎓 Galgotias AI Assistant (Advanced)")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("API key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

@st.cache_resource(show_spinner="Waking up the AI and loading data...")
def initialize_backend():
    return get_rag_chain()

rag_chain = initialize_backend()

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # 1. Translate Streamlit history into LangChain history objects
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            # We strip out the references so the LLM doesn't get confused by its own formatting
            clean_content = msg["content"].split("\n\n**References:**")[0]
            chat_history.append(AIMessage(content=clean_content))
    
    # Only keep the last 6 exchanges (12 messages) to avoid token overflow
    chat_history = chat_history[-12:]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 2. Pass BOTH the new query and the chat history to the chain
                response = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": chat_history
                })
                
                answer = response["answer"]
                sources = set([doc.metadata.get('source', 'Unknown source') for doc in response["context"]])
                source_text = "\n\n**References:**\n" + "\n".join([f"- {url}" for url in sources])
                
                full_response = answer + source_text
                st.markdown(full_response)
            except Exception as e:
                full_response = "Sorry, something went wrong. Please try again."
                st.error(full_response)
            
    # Save the updated conversation to Streamlit memory
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": full_response})