import os
import streamlit as st
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="ScholarMind Briefing Agent", page_icon="🔬", layout="wide")

st.title("🔬 ScholarMind: Research Paper Briefing Agent")
st.caption("Developed by Arav | Accelerated Learning & Paper Analysis Engine")

with st.sidebar:
    st.header("📄 Document Hub")
    secret_key = st.secrets.get("GOOGLE_API_KEY", "")
    if secret_key:
        api_key = secret_key
    else:
        api_key = st.text_input("Enter Google API Key", type="password")
        
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])
    st.divider()
    st.caption("Powered by LangChain, FAISS & Gemini")

@st.cache_resource(show_spinner=False)
def process_pdf(_file):
    with open("temp.pdf", "wb") as f:
        f.write(_file.read())
        
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return docs, vectorstore

def generate_paper_brief(full_text, llm):
    prompt = f"""You are an expert academic briefing agent. Analyze the following research paper text and provide a structured presentation-ready executive brief.

Sections required:
1. 🎯 **Core Claims & Contributions**: What are the main novel ideas/claims?
2. 🔬 **Methodology**: How did the authors test or prove their claims?
3. 📈 **Key Findings**: What were the primary results?
4. ⚠️ **Limitations & Critiques**: What limitations or weaknesses did the authors mention or leave out?

Paper Text Excerpt:
{full_text[:8000]}"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)

def generate_flashcards(full_text, llm):
    prompt = f"""Generate 5 high-value study flashcards from this paper. 
Return ONLY a raw JSON list of objects, where each object has "question" and "answer" keys.
Do not include markdown code block formatting (no ```json).

Paper Text Excerpt:
{full_text[:6000]}"""
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)
    try:
        clean_content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_content)
    except:
        return [
            {"question": "Core Topic", "answer": "Refer to the paper executive brief tab for main details."},
            {"question": "Methodology", "answer": "Refer to the methodology section in the brief."}
        ]

def generate_answer(query, retriever, llm):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Use the provided context to answer the user query strictly grounded in the document.

Context:
{context}

Query:
{query}"""
    
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)
    return content, docs

if uploaded_file and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    with st.spinner("Indexing research paper & building agent intelligence..."):
        try:
            docs, vectorstore = process_pdf(uploaded_file)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
            st.sidebar.success("Paper Analyzed Successfully!")
        except Exception as e:
            st.error(f"Error initializing pipeline: {e}")
            st.stop()

    full_text = " ".join([d.page_content for d in docs])

    tab1, tab2, tab3 = st.tabs(["📊 Executive Brief", "🃏 Study Flashcards", "💬 Interactive QA"])

    with tab1:
        st.subheader("📋 Presentation-Ready Paper Brief")
        with st.spinner("Extracting claims, methodology & limitations..."):
            brief = generate_paper_brief(full_text, llm)
            st.markdown(brief)

    with tab2:
        st.subheader("🧠 Concept Flashcards")
        with st.spinner("Generating flashcards..."):
            flashcards = generate_flashcards(full_text, llm)
            for i, card in enumerate(flashcards):
                with st.expander(f"🎴 Card {i+1}: {card.get('question', 'Question')}"):
                    st.write(f"**Answer:** {card.get('answer', 'Answer')}")

    with tab3:
        st.subheader("🔍 Grounded QA with Citation Tracking")
        user_query = st.text_input("Ask a specific question about this paper:")
        if user_query:
            with st.spinner("Retrieving cited evidence..."):
                answer, source_docs = generate_answer(user_query, retriever, llm)
                st.markdown("### 💡 Answer")
                st.write(answer)
                
                with st.expander("🔎 View Source Text Citations"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Citation Chunk {i+1}:**")
                        st.info(doc.page_content)

elif not api_key:
    st.warning("Please enter your Google API Key in the sidebar to begin.")
elif not uploaded_file:
    st.info("Please upload a PDF research paper in the sidebar.")