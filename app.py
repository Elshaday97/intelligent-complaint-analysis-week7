import streamlit as st
import sys
import os

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.rag_system import RAGSystem

# Page Configuration
st.set_page_config(page_title="Financial Support AI", layout="wide")

st.title("Financial Complaint Analysis AI")
st.markdown(
    """
This tool uses **Retrieval-Augmented Generation (RAG)** to answer questions 
based on thousands of real financial complaints.
"""
)


# 2. Load the RAG System (Cached)
# We use cache_resource because the object is complex/heavy (database connection)
@st.cache_resource
def load_rag():
    return RAGSystem()


try:
    with st.spinner("Loading Knowledge Base..."):
        rag = load_rag()
    st.success("System Ready!")
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about overdrafts, credit reporting, etc..."):
    # A. Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & Generating answer..."):

            # 1. Retrieve
            retrieved_docs = rag.search_vector_db(prompt)

            # 2. Generate (Calls your LLM)
            response_text = rag.agument_result(prompt, retrieved_docs)

            st.markdown(response_text)

            # Optional: Show Evidence (Collapsible)
            with st.expander("View Retrieved Source Context"):
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(
                        f"**Source {i+1}** (Product: {doc.metadata.get('product', 'Unknown')})"
                    )
                    st.caption(doc.page_content)

    # C. Save AI Message to History
    st.session_state.messages.append({"role": "assistant", "content": response_text})
