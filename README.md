# 🎓 Galgotias University AI Chatbot

A RAG (Retrieval-Augmented Generation) based chatbot for Galgotias University, built with LangChain, Streamlit, and Groq.

## Features

- Automatically crawls the university website to build a knowledge base
- Answers questions about admissions, programs, accreditations, and more
- Remembers conversation history for follow-up questions
- References only shown when explicitly requested by the user
- Runs fully locally for embeddings (no embedding API costs)

## Tech Stack

| Component    | Technology                                                   |
| ------------ | ------------------------------------------------------------ |
| UI           | Streamlit                                                    |
| LLM          | Groq (Llama 3.3 70B)                                         |
| Embeddings   | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector Store | ChromaDB (persistent on disk)                                |
| Web Crawling | LangChain `RecursiveUrlLoader`                               |
| Framework    | LangChain                                                    |

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd uniChatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv myvenv
# Windows
myvenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Set up your API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com/keys

### 5. Run the app

```bash
streamlit run app.py
```

On the first run, the app will automatically crawl the Galgotias University website and build the vector database. This takes a few minutes. Subsequent runs load instantly from disk.

## Project Structure

```
uniChatbot/
├── app.py              # Streamlit UI and chat logic
├── dataIngestion.py    # Web crawling, embeddings, and RAG chain
├── requirement.txt     # Python dependencies
├── .env                # API keys (do not commit)
├── chroma_db/          # Persistent vector database (auto-generated)
└── myvenv/             # Virtual environment
```

## Resetting the Knowledge Base

If you want to re-crawl the website (e.g., after the site is updated), delete the `chroma_db` folder and restart:

```bash
rmdir /s /q chroma_db
streamlit run app.py
```

## Notes

- The `.env` file and `chroma_db/` folder are not committed to Git
- The chatbot only answers questions related to Galgotias University
- Ask "show me the sources" or "give me the reference" to see source URLs
