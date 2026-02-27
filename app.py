import streamlit as st
from dataIngestion import get_rag_chain # <-- Importing your backend function!

# 1. Set up the Streamlit UI configuration
st.set_page_config(page_title="Galgotias University Bot", page_icon="🎓")
st.title("🎓 Galgotias University AI Assistant")
st.write("Ask me anything about the university, its vision, or its core values!")

# 2. Initialize the backend exactly once
@st.cache_resource(show_spinner="Waking up the AI and loading data...")
def initialize_backend():
    return get_rag_chain()

rag_chain = initialize_backend()

# 3. Manage chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle user input
user_query = st.chat_input("Ask a question...")

if user_query:
    # Display user's question immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    # Save it to history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate and display the bot's response
    with st.chat_message("assistant"):
        with st.spinner("Searching the university database..."):
            
            # Call the backend chain
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            
            # Format the sources cleanly
            sources = set([doc.metadata.get('source', 'Unknown source') for doc in response["context"]])
            source_text = "\n\n**References:**\n" + "\n".join([f"- {url}" for url in sources])
            
            full_response = answer + source_text
            st.markdown(full_response)
            
    # Save the bot's full response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})