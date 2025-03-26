import streamlit as st
import os
import tempfile
import gc
import base64
import time
import uuid

from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from typing import Type, List, Any
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
# Define Agents & Tasks
# ---------------------------
def create_agents_and_tasks(pdf_tool: BaseTool):
    web_search_tool = FireCrawlWebSearchTool()

    retriever_agent = Agent(
        role="Retrieve relevant information to answer the user query: {query}",
        goal=(
            "Retrieve the most relevant information from the available sources "
            "for the user query: {query}. Always try to use the PDF search tool first. "
            "If you are not able to retrieve the information from the PDF search tool, "
            "then try to use the web search tool."
        ),
        backstory=(
            "You're a meticulous analyst with a keen eye for detail. "
            "You're known for your ability to understand user queries: {query} "
            "and retrieve knowledge from the most suitable knowledge base."
        ),
        verbose=True,
        tools=[pdf_tool, web_search_tool],
        llm=load_llm()
    )

    response_synthesizer_agent = Agent(
        role="Response synthesizer agent for the user query: {query}",
        goal=(
            "Synthesize the retrieved information into a concise and coherent response "
            "based on the user query: {query}. If you are not able to retrieve the "
            "information then think harder and think creatively of how the user question could be linked to the document source or the source pulled from the internet,"
            "if you truly don't have that information, "
            " respond with \"I'm sorry, I couldn't find the information "
            "you're looking for.\""
        ),
        backstory=(
            "You're a skilled communicator with a knack for turning "
            "complex information into clear and concise responses."
        ),
        verbose=True,
        llm=load_llm()
    )

    retrieval_task = Task(
        description=(
            "Retrieve the most relevant information from the available "
            "sources for the user query: {query}"
        ),
        expected_output=(
            "The most relevant information in the form of text as retrieved "
            "from the sources."
        ),
        agent=retriever_agent
    )

    response_task = Task(
        description="Synthesize the final response for the user query: {query}",
        expected_output=(
            "A detailed, coherent, and elaborated response based on the retrieved information "
            "from the right source for the user query: {query}. If you are not "
            "able to retrieve the exact information, think about how the user question might relate to the ingested source,"
            "If you still cannot come up with a reliable answer, fall back to using the webscraping tool, but you must"
            "explicitly say that you've utilized webscraping to retrieve the information to answer the user's question,"
            "if the user explicitly forbids you to use webscraping, go back to the document source and see if you can come up with an answer,"
            "if not,"
            "then respond with: "
            "\"I'm sorry, I couldn't find the information you're looking for.\""
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
        lines = result.split('\n')
        for i, line in enumerate(lines):
            full_response += line
            if i < len(lines) - 1:
                full_response += '\n'
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.15)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": result})
