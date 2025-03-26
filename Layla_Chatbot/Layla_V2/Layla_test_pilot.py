import streamlit as st
import os
import tempfile
import gc
import base64
import time
import uuid

from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from typing import List, Any
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from markitdown import MarkItDown
from dotenv import load_dotenv
from pathlib import Path

# ---------------------------
# Load Environment Variables
# ---------------------------
env_path = "../../.env"
load_dotenv(dotenv_path=env_path)
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")

import openai

# ---------------------------
# Initialize Pinecone with the New API
# ---------------------------
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=pinecone_api_key)
serverless_spec = ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)

# ---------------------------
# Integrated Custom Tools
# ---------------------------
class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    index_name: str = "layla-collection"  
    dimension: int = 1536  

    _file_path: str = PrivateAttr()
    _index: Any = PrivateAttr()

    def __init__(self, file_path: str):
        super().__init__()
        self._file_path = file_path  
        
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=serverless_spec
            )
        self._index = pc.Index(self.index_name)
        self._process_document()

    def _extract_text(self) -> str:
        md = MarkItDown()
        result = md.convert(self._file_path)
        return result.text_content

    def _get_openai_embedding(self, text: str) -> List[float]:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def _create_chunks(self, raw_text: str) -> List[str]:
        chunk_size = 512
        return [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)]

    def _process_document(self):
        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)
        vectors = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            embedding = self._get_openai_embedding(chunk)
            if not embedding or len(embedding) != self.dimension:
                continue
            point_id = str(uuid.uuid4())
            vectors.append({
                "id": point_id,
                "values": embedding,
                "metadata": {"text": chunk}
            })
        if vectors:
            self._index.upsert(vectors=vectors)
        else:
            st.error("No valid vectors to upsert!")

    def _run(self, query: str) -> str:
        query_embedding = self._get_openai_embedding(query)
        response = self._index.query(vector=query_embedding, top_k=5, include_metadata=True)
        docs = [match["metadata"]["text"] for match in response["matches"]]
        return "\n___\n".join(docs)

class FireCrawlWebSearchTool(BaseTool):
    name: str = "FireCrawlWebSearchTool"
    description: str = "A placeholder web search tool."

    def _run(self, query: str) -> str:
        return f"Web search result for query: {query}"

# ---------------------------
# LLM Loading (Cached)
# ---------------------------
@st.cache_resource
def load_llm():
    llm = LLM(
        model="gpt-4o",      
        verbose=True,
        temperature=1,
    )
    return llm

# ---------------------------
# Define Agents & Tasks with Fine-Tuned Prompts
# ---------------------------
def create_agents_and_tasks(pdf_tool: BaseTool):
    web_search_tool = FireCrawlWebSearchTool()

    # Fine-tuned Retriever Agent: Work hard to extract detailed information.
    retriever_agent = Agent(
        role="Data Retrieval Agent for Document and Web Sources",
        goal=(
            "For the given user query: {query}, extract all available detailed and relevant information. "
            "If the query is clearly related to the uploaded PDF document, exhaustively search the document to extract every pertinent detail. "
            "If the query appears unrelated or if the document does not provide sufficient data, perform an in-depth web search to gather supplementary information. "
            "Do not default to saying 'I don't have the information' – always try to retrieve and produce the most comprehensive result possible."
        ),
        backstory=(
            "You are a highly skilled information retrieval expert with deep expertise in document analysis and web data acquisition. "
            "Your task is to search both the uploaded PDF and, when needed, perform web searches to gather exhaustive, detailed, and accurate information. "
            "You always strive to extract every relevant detail and never give up until you provide a complete answer."
        ),
        verbose=True,
        tools=[pdf_tool, web_search_tool],
        llm=load_llm()
    )

    # Fine-tuned Response Synthesizer Agent: Produce detailed, comprehensive answers.
    response_synthesizer_agent = Agent(
        role="Response Synthesizer and Detail Enhancer",
        goal=(
            "Synthesize the retrieved information into a comprehensive, detailed, and accurate final response for the query: {query}. "
            "Integrate all retrieved data from the document and/or the web. If some aspects of the query are not directly covered by the document, intelligently incorporate relevant web information. "
            "Ensure that every part of the query is answered thoroughly and with rich detail, and avoid any vague or incomplete responses."
        ),
        backstory=(
            "You are an expert communicator with a talent for transforming raw data into detailed and coherent narratives. "
            "Your role is to carefully integrate the detailed information gathered by the retrieval agent into an all-encompassing answer that leaves no aspect of the query unanswered."
        ),
        verbose=True,
        llm=load_llm()
    )

    retrieval_task = Task(
        description=(
            "Retrieve the most comprehensive and detailed relevant information for the query: {query}. "
            "Prioritize extracting all possible details from the PDF document. If the document is insufficient, perform a detailed web search. "
            "Your response should be exhaustive and not simply a summary."
        ),
        expected_output=(
            "A detailed and thorough retrieval of information from the document and/or the web that covers every aspect of the query."
        ),
        agent=retriever_agent
    )

    response_task = Task(
        description=(
            "Synthesize a final answer for the user query: {query} by integrating the detailed information retrieved. "
            "If any information is missing from the document, incorporate relevant web data and make thoughtful inferences. "
            "Ensure that the final response is comprehensive, detailed, and leaves no aspect of the query unaddressed."
        ),
        expected_output=(
            "A final, elaborated response that fully addresses the query with rich details and thorough explanations, including any necessary citations or references to the source data."
        ),
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True
    )
    return crew

# ---------------------------
# Streamlit Setup
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None 

if "crew" not in st.session_state:
    st.session_state.crew = None      

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def display_pdf(file_bytes: bytes, file_name: str):
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="600px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

with st.sidebar:
    st.header("Add Your PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.session_state.pdf_tool is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                with st.spinner("Indexing PDF... Please wait..."):
                    st.session_state.pdf_tool = DocumentSearchTool(file_path=temp_file_path)
            st.success("PDF indexed! Ready to chat.")
        display_pdf(uploaded_file.getvalue(), uploaded_file.name)
    st.button("Clear Chat", on_click=reset_chat)

st.markdown("""
    # Layla Chat Bot Version .2 (Document search + Webscrape fall back)
""")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about your PDF...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if st.session_state.crew is None:
        st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            inputs = {"query": prompt}
            result = st.session_state.crew.kickoff(inputs=inputs).raw
        for i, line in enumerate(result.split('\n')):
            full_response += line + ("\n" if i < len(result.split('\n')) - 1 else "")
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.15)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": result})
