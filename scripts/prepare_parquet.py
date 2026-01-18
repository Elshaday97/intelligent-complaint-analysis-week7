import sys
import os
from pathlib import Path
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added project root to path: {project_root}")

from scripts.constants import (
    EMBEDDED_COMPLAINTS_FILE_PATH,
    VECTOR_STORE_PATH,
    Embedding_Columns,
)

OUTPUT_PATH = os.path.join(VECTOR_STORE_PATH, "embedded")


def main():
    print(f"Loading data from {EMBEDDED_COMPLAINTS_FILE_PATH}...")

    if not os.path.exists(EMBEDDED_COMPLAINTS_FILE_PATH):
        print(f"Error: File not found at {EMBEDDED_COMPLAINTS_FILE_PATH}")
        return

    df = pd.read_parquet(EMBEDDED_COMPLAINTS_FILE_PATH)

    # --- 1. Prepare Data for FAISS ---
    text_embeddings = []
    metadatas = []

    doc_col = Embedding_Columns.DOCUMENT.value

    for index, row in df.iterrows():
        text = row[doc_col]
        vector = row[Embedding_Columns.EMBEDDING.value]

        # Ensure metadata is a dict
        metadata = (
            row[Embedding_Columns.METADATA.value]
            if Embedding_Columns.METADATA.value in df.columns
            else {}
        )
        if not isinstance(metadata, dict):
            metadata = {}

        text_embeddings.append((text, vector))
        metadatas.append(metadata)
        complaint_id = row.get(Embedding_Columns.ID.value, str(index))
        metadata[Embedding_Columns.ID.value] = complaint_id

    # --- 2. Initialize the Embedding Object ---
    print(" Loading embedding model wrapper...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- 3. Build the Index ---
    print(f"Building FAISS index from {len(text_embeddings)} pre-computed vectors...")
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings, embedding=embedding_model, metadatas=metadatas
    )

    # --- 4. Save to Disk ---
    print(f"Saving index to '{OUTPUT_PATH}'...")
    vector_store.save_local(OUTPUT_PATH)

    print("Done! You can now run your RAG system.")


if __name__ == "__main__":
    main()
