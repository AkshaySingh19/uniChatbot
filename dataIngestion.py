import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

def get_rag_chain():
    """Initializes the database, LLM, and returns the functional RAG chain."""
    
    persist_directory = "./chroma_db"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    if os.path.exists(persist_directory):
        print("Found existing database! Loading from disk...")
        vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
    else:
        print("No database found. Scraping website and creating new database...")
        urls = [
            "https://www.galgotiasuniversity.edu.in/",
            "https://www.galgotiasuniversity.edu.in/about-us",
            "https://www.galgotiasuniversity.edu.in/p/about/vision-and-mission"
        ]
        loader = WebBaseLoader(web_paths=urls)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    system_prompt = (
        "You are a helpful assistant for Galgotias University. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer, just say that you don't know. "
        "Always provide your answer clearly and concisely.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain