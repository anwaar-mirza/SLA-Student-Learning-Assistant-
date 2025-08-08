# 🧠 Student Learning Assistant

A conversational assistant that helps students understand and explore the contents of uploaded PDF documents using a smart question-answering system built with LangChain, Groq, FAISS, and Hugging Face Embeddings.

---

## 🚀 Features

- 🔍 **PDF-based Question Answering:** Upload any PDF and ask questions about its contents.
- 🧠 **RAG Architecture:** Uses Retrieval-Augmented Generation to provide context-aware, accurate answers.
- 💬 **Session-based Memory:** Keeps track of chat history using Streamlit session state.
- ⚡ **Groq LLM Integration:** Uses `gemma2-9b-it` model from Groq for fast and powerful responses.
- 🧩 **Semantic Chunking & Search:** Documents are split into overlapping chunks and indexed using FAISS for effective retrieval.
- 📎 **Chat Interface:** Clean and interactive chat layout using Streamlit.

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit  
- **LLM Provider:** Groq (gemma2-9b-it`)  
- **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)  
- **Vector Store:** FAISS  
- **PDF Parsing:** LangChain PyPDFLoader  
- **Memory:** LangChain ChatMessageHistory with Streamlit session management

---

## 📁 File Structure

```bash
.
├── StudentAssistan.py         # Main Streamlit application
├── essential.py               # Stores `contextualize_q_system_prompt` and `chat_prompt_template`                     
├── requirements.txt           # Python dependencies
└── README.md                  # This file



