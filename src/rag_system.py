import sys
import os
from pathlib import Path
from dotenv import load_dotenv

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import HumanMessage
from scripts.constants import Embedding_Columns

load_dotenv()


class RAGSystem:
    def __init__(self):
        print("Initializing RAG System")
        self.vector_store_path = os.path.join(project_root, "vector_store", "embedded")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_db = FAISS.load_local(
            self.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        endpoint = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="conversational",
            temperature=0.5,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )

        self.llm = ChatHuggingFace(llm=endpoint)

        print("RAG System Ready!")

    def initiate_chat(self):
        while True:
            try:
                # Take input from user
                user_query = input("How can I help you today?\n")
            except KeyboardInterrupt:
                print("\n Exiting..")
                break

            try:
                # Vectorized Search
                retrieved_docs = self.search_vector_db(user_query)

                # Get Augumented Results
                response = self.agument_result(user_query, retrieved_docs)

                print("\n--- AI Response ---")
                print(response)

                print("\n--- Sources ---")

                for i, doc in enumerate(retrieved_docs):
                    full_metadata = doc.metadata.get(
                        Embedding_Columns.METADATA.value, "Unknown Product"
                    )
                    comp_id = doc.metadata.get(Embedding_Columns.ID.value, "N/A")

                    # Print a clean summary
                    print(full_metadata)
                    print(f"Source {i+1} (ID: {comp_id})")
                    print(f'   "{doc.page_content[:150]}..."')  # Print first 150 chars
                    print("")  # Empty line for spacing
                    print("-----------------------")

            except Exception as e:
                print(f"\nError: {e}")

    def search_vector_db(self, user_query: str):
        # Search user query in vector database
        results = self.vector_db.similarity_search(
            user_query, k=5
        )  # vectorization of user input is handled behind the scene
        return results

    def agument_result(self, user_query: str, context_docs):
        # Prepare context & prompt
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt_template = f"""
        You are a financial analyst assistant for CrediTrust. Your task is to answer questions
        about customer complaints. Use the following retrieved complaint excerpts to
        formulate your answer. If the context doesn't contain the answer, state that you don't
        have enough information.
        
        CONTEXT:
        {context_text}

        QUESTION:
        {user_query}

        ANSWER:
        """

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        formatted_prompt = prompt.format(context=context_text, question=user_query)

        # Generate Response
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        return response.content


rag_sys = RAGSystem()
rag_sys.initiate_chat()
