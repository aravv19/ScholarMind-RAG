# 🔬 ScholarMind: AI Document Assistant

ScholarMind is a Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, **FAISS**, and **Google Gemini**. It allows users to upload PDF research papers or documents and ask questions to receive precise, grounded answers accompanied by verifiable source citations.

---

## ✨ Features

- **📄 Document Upload:** Support for PDF file uploads via an intuitive Streamlit UI.
- **⚡ Fast Vector Search:** Uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) and `FAISS` for quick local semantic search.
- **🧠 Grounded Insights:** Leverages Google's `gemini-flash-latest` model with strict context-bounding to prevent hallucinations.
- **🔍 Source Traceability:** View the exact retrieved chunks of context used by the model to construct each answer.
- **💬 Chat Memory:** Maintains message history during your session for a natural conversational experience.

---

## 🛠️ Tech Stack

- **UI Framework:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **LLM Engine:** Google Gemini API (`gemini-flash-latest`)

---

## 📦 Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/aravv19/scholarmind.git](https://github.com/your-username/scholarmind.git)
cd scholarmind
