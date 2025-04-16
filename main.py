import os
import openai
import streamlit as st
from openai import OpenAI
from langchain.document_loaders import TextLoader
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as LangchainOpenAI

# ✅ Load DeepSeek API credentials
DEEPSEEK_API_KEY = "sk-4641fc1a8560499da62c883dd3472e0c"
DEEPSEEK_API_BASE = "https://api.deepseek.com"

# # ✅ Ensure API key is set
# if not DEEPSEEK_API_KEY or "sk-4641fc1a8560499da62c883dd3472e0c" in DEEPSEEK_API_KEY:
#     raise ValueError("❌ DeepSeek API Key is missing! Set 'DEEPSEEK_API_KEY' in your environment.")

# ✅ Initialize DeepSeek OpenAI client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)

# ✅ Initialize Embeddings (HuggingFace)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Function to Build FAISS Index
def build_faiss_index():
    """Builds the FAISS index from documents."""
    if not os.path.exists("sample_docs.txt"):
        raise FileNotFoundError("❌ 'sample_docs.txt' not found! Please add your documents.")

    loader = TextLoader("sample_docs.txt")
    docs = loader.load()
    vector_store = FAISS.from_documents(docs, embeddings)

    os.makedirs("faiss_index", exist_ok=True)
    vector_store.save_local("faiss_index")
    print("✅ FAISS index successfully built and saved.")

# ✅ Function to Load FAISS Index
@st.cache_resource
def load_vector_store():
    """Loads FAISS index if available, otherwise raises an error."""
    if not os.path.exists("faiss_index"):
        raise FileNotFoundError("❌ FAISS index not found! Please build the index first.")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# ✅ Build FAISS Index if needed
if __name__ == "__main__":
    build_faiss_index()

# ✅ Initialize Streamlit App
st.title("RAG System with Streamlit - DeepSeek API")

# ✅ Load vector store and initialize retriever
vector_store = load_vector_store()
retriever = vector_store.as_retriever()

# ✅ Initialize RAG-based QA system
qa_chain = RetrievalQA.from_chain_type(llm=LangchainOpenAI(openai_api_key=DEEPSEEK_API_KEY), retriever=retriever)

# ✅ Streamlit UI for user input
query = st.text_input("Enter your query:")

if query:
    # 🔥 Retrieve relevant documents using FAISS
    response = qa_chain.run(query)

    # 🔥 Use DeepSeek API for chat-based responses
    chat_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": response},
        ],
        stream=False
    )

    st.write("📖 Retrieved Answer:", response)
    st.write("💬 DeepSeek Response:", chat_response.choices[0].message.content)
