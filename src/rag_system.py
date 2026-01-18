from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from scripts.constants import EMBEDDED_COMPLAINTS_FILE_PATH
import os
from dotenv import load_dotenv

load_dotenv()


class RAGSystem:
    def __init__(self):
        print("Initializing RAG System")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_db = FAISS.load_local(
            EMBEDDED_COMPLAINTS_FILE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.5,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        print("RAG System Ready!")

    def initiate_chat(self):
        # Take input from user
        user_query = input("How can I help you today?")

        # Vectorized Search
        retrieved_docs = self.search_vector_db(user_query)

        # Get Augumented Results
        response = self.agument_result(user_query, retrieved_docs)

        print("\n--- AI Response ---")
        print(response)

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
        response = self.llm.invoke(formatted_prompt)

        return response
