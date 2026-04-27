# 🚀 RAG-Based AI Document Question Answering System

An AI-powered document question-answering system built using **Retrieval-Augmented Generation (RAG)**.  
Upload documents and ask natural language questions to get **context-aware answers**.

---

## 📌 Features

- 📄 Upload PDF and TXT documents
- 🔍 Semantic search using embeddings
- ⚡ Fast retrieval using FAISS (vector database)
- 🤖 Context-aware answers using LLM (Llama 3 via Groq)
- 📊 Response metrics (latency, similarity score)
- 🌐 Simple frontend for interaction

---

## 🧠 Tech Stack

- **Backend:** FastAPI  
- **Vector Database:** FAISS  
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)  
- **LLM:** Llama 3 (Groq API)  
- **Frontend:** HTML, CSS, JavaScript  

---

## 🔍 How It Works (RAG Pipeline)

1. Upload document (PDF/TXT)
2. Extract text from file
3. Split text into chunks
4. Convert chunks into embeddings
5. Store embeddings in FAISS
6. User asks a question
7. Retrieve most relevant chunks
8. Send context + question to LLM
9. Generate accurate answer

---

## 📂 Project Structure


rag_api/
│── main.py
│── routes/
│── services/
│── utils/
│── vector_store/
│── models/
│── front.html
│── requirements.txt


---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Yuvrajkumar46/RAG-Based-AI-Documentation.git
cd RAG-Based-AI-Documentation
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file and add:

GROQ_API_KEY=your_api_key_here
LLM_MODEL=llama3-70b-8192
5. Run the server
uvicorn main:app --reload
6. Access the application
API Docs: http://localhost:8000/docs
Frontend: Open front.html
📊 Example Use Case
Upload a resume → Ask: “What skills are mentioned?”
Upload research paper → Ask: “Summarize the findings”
Upload job description → Ask: “Required technologies?”
⚠️ Limitations
LLM depends on external API (Groq)
Retrieval quality depends on chunking
May fail on very complex documents
🚀 Future Improvements
Chat history (conversation memory)
Better UI (ChatGPT-style interface)
Streaming responses
Multi-document filtering
Deployment (Render / Railway)
🎯 Key Learning
Implemented end-to-end RAG pipeline
Improved LLM accuracy using retrieval
Reduced hallucination with contextual grounding
Built scalable API using FastAPI
📌 Author

Yuvraj Kumar Jaiswal
