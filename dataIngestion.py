import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_rag_chain():
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(persist_directory):
        print("Loading existing database from disk...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("No database found. Crawling website recursively...")
        
        def bs4_extractor(html: str) -> str:
            """Extract clean text from HTML using BeautifulSoup."""
            soup = BeautifulSoup(html, "html.parser")
            # Remove script, style, nav, footer elements
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Remove excessive blank lines
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Remove URLs from text
            text = re.sub(r'https?://\S+', '', text)
            return text

        base_url = "https://www.galgotiasuniversity.edu.in/"
        
        loader = RecursiveUrlLoader(
            url=base_url,
            max_depth=5,              
            extractor=bs4_extractor,  
            prevent_outside=True,     
            timeout=30,
        )
        docs = loader.load()
        
        # Filter out empty or very short pages
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 100]
        
        print(f"Crawled {len(docs)} pages from the website.")

        for doc in docs:
            doc.metadata["source_title"] = doc.metadata.get("title", doc.metadata.get("source", "Unknown"))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=250)
        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=persist_directory
        )
        
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 4, "fetch_k": 20}
    )
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are a helpful assistant for Galgotias University. "
        "ONLY answer questions related to Galgotias University. "
        "If the question is not about the university, politely say you can only help with university-related queries. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't find the answer in the context, say you don't know — never make up information. "
        "Always provide your answer clearly and concisely.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain