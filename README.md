# 🔬 ScholarMind: AI Document Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://scholarmind-rag-1910.streamlit.app)

> **ScholarMind** is an intelligent Document QA application that allows researchers, students, and professionals to upload complex PDF documents and instantly query them for grounded insights with exact source citations.

🔗 **Live Demo:** [Click here to launch ScholarMind](https://scholarmind-rag-1910.streamlit.app)

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

## 📦 Local Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/aravv19/scholarmind.git](https://github.com/aravv19/scholarmind.git)
cd scholarmind
