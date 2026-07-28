import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Page Config
st.set_page_config(page_title="ScholarMind RAG", page_icon="🔬", layout="wide")

# App Header
st.title("🔬 ScholarMind: AI Document Assistant")
st.caption("Developed by Arav | Powered by LangChain, FAISS & Gemini")
st.markdown("Upload any research paper or document to extract grounded insights with source citations.")

# Sidebar - Document Uploader
with st.sidebar:
    st.header("📄 Document Hub")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    st.divider()
    

# Retrieve API Key automatically from Secrets or Environment
api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

@st.cache_resource(show_spinner=False)
def process_pdf(_file):
    with open("temp_paper.pdf", "wb") as f:
        f.write(_file.read())
        
    loader = PyPDFLoader("temp_paper.pdf")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

def generate_answer(query, retriever, llm):
    docs = retriever.invoke(query)
    context = "\n\n".join([f"[Source Chunk {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])
    
    prompt = f"""You are an expert research assistant. 
Answer the query using ONLY the provided context excerpts. If the answer is not contained in the context, explicitly state "Information not present in the uploaded document."

Context Excerpts:
{context}

Query:
{query}"""
    
    response = llm.invoke(prompt)
    
    if hasattr(response, 'content'):
        content = response.content
        if isinstance(content, str):
            return content, docs
        elif isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            return content[0].get('text', str(content)), docs
    return str(response), docs

# Main App Execution
if not api_key:
    st.error("⚠️ API Key not configured. Please check .streamlit/secrets.toml file.")
elif uploaded_file:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    with st.spinner("Indexing document & building vector database..."):
        try:
            vectorstore = process_pdf(uploaded_file)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
            st.sidebar.success("✅ Document Indexed!")
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("🔍 View Referenced Context Chunks"):
                    for i, doc in enumerate(message["sources"]):
                        st.info(f"**Chunk {i+1}:**\n\n{doc.page_content}")

    if user_query := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                answer, source_docs = generate_answer(user_query, retriever, llm)
                st.markdown(answer)
                
                with st.expander("🔍 View Referenced Context Chunks"):
                    for i, doc in enumerate(source_docs):
                        st.info(f"**Chunk {i+1}:**\n\n{doc.page_content}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": source_docs
                })

else:
    st.info("👈 Upload a PDF document in the sidebar to start asking questions!")
