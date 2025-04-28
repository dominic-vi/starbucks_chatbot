## Starbucks's RAG Chatbot
RAG Chatbot designed to answer questions about Starbucks products and services by leveraging Retrieval-Augmented Generation (RAG) techniques.

### 🚀 Features
> Answers user queries about Starbucks menus, promotions, and store policies.

> Retrieval-Augmented Generation (RAG) system for more accurate, context-aware responses.

> Embedding model for efficient information retrieval.

### 🛠️ Technologies Used
> Ollama for hosting the Language Model

> LangChain for building the RAG system

> FAISS for vector storage

> halong-embedding model for text embeddings

> ViRanker for reranking results

> ChainLit for creating the frontend interface

### 📦 Installation
1. Clone the repository:
    git clone https://github.com/dominic-vi/starbucks_chatbot
    
2. Install the required dependencies:
    pip install -r requirements.txt

3. Initialize the vector database:
    python ini_vector_db.py

4. Start the application:
    chainlit run app.py
