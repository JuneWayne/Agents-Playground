import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

pdf_files = [
    'Data_file/ACLR_Research.pdf',
    'Data_file/ChildVaccine.pdf'
]

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

for pdf_file in pdf_files:
    print(f"--- Processing {pdf_file} ---")
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()  

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"{pdf_file} => Created {len(chunks)} chunks")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.environ.get("INDEX_NAME"))
