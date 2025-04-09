import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

load_dotenv('../../.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Chatbot:
    def __init__(self):
        self.pdf_path = '../../Data_file/ACLR_Research.pdf'
        self.vectorestore = self._create_vector_db()
    def _create_vector_db(self):
        split_docs = self.extract_text_from_pdf()
        embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
        vectorstore = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory="chroma_db")
        return vectorstore
    def extract_text_from_pdf(self):
        loader = PyPDFLoader(self.pdf_path, mode='single')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)
        return split_docs
    def generate_response(self, query):
        docs = self.vectorestore.similarity_search(query)
        context = ""
        for i, doc in enumerate(docs):
            context += f"{i+1}. {doc.page_content}\n"
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {   "role": "system", "content": "You are a helpful research assistant who is excellent at helping researchers find relevant information in their documents."},
                {   "role": "assistant", "content": "I can help you find relevant information in your documents. Please provide me with a query."},
                {   "role": "user", "content": f"Given the following context, answer the question: {query}\n\nContext:\n{context}"}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content, context
    