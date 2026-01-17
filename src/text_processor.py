from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from scripts.constants import Columns
from langchain_core.documents import Document


class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_text(self, text) -> list:
        """
        Splits the input text into chunks of specified size.
        Returns:
            list: A list containing the text chunks.
        """
        return self.splitter.split_text(text)

    def split_documents(self, df: pd.DataFrame) -> list:
        docs = []
        for index, row in df.iterrows():
            text = row[Columns.COMPLAINT.value]
            chunks = self.split_text(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "id": row[Columns.COMPLAINT_ID.value],
                        "product": row[Columns.PRODUCT.value],
                    },
                )
                docs.append(doc)

        return docs
