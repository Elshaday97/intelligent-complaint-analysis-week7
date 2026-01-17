from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class VectorManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None

    def create_vector_store(self, documents):
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def save_vector_store(self, path="vector_store/"):
        self.vector_store.save_local(path)
