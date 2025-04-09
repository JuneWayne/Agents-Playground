import streamlit as st
import os
import tempfile
import gc
import base64
import time
import uuid
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from markitdown import MarkItDown

from pinecone import Pinecone, ServerlessSpec

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_correctness, context_recall, context_precision

load_dotenv("../../.env")
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")

critic_llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
openai_embeddings = OpenAIEmbeddings(api_key=openai_api_key)

pc = Pinecone(api_key=pinecone_api_key)
serverless_spec = ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    index_name: str = "layla-collection"
    dimension: int = 1536

    def __init__(self, file_path: str):
        super().__init__()
        self._file_path = file_path
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(name=self.index_name, dimension=self.dimension, metric="cosine", spec=serverless_spec)
        self._index = pc.Index(self.index_name)
        self._process_document()

    def _extract_text(self) -> str:
        md = MarkItDown()
        result = md.convert(self._file_path)
        return result.text_content

    def _get_openai_embedding(self, text: str):
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        return response.data[0].embedding

    def _create_chunks(self, raw_text: str):
        chunk_size = 512
        return [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]

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

    def _run(self, query: str):
        query_embedding = self._get_openai_embedding(query)
        response = self._index.query(vector=query_embedding, top_k=5, include_metadata=True)
        docs = [match["metadata"]["text"] for match in response["matches"]]
        return {"retrieved_text": "\n___\n".join(docs), "chunks": docs}

class FireCrawlWebSearchTool(BaseTool):
    name = "FireCrawlWebSearchTool"
    description = "A placeholder web search tool."
    def _run(self, query: str) -> str:
        return f"Web search result for query: {query}"

@st.cache_resource
def load_llm():
    llm = LLM(model="gpt-4o", verbose=True, temperature=1)
    return llm

def create_agents_and_tasks(pdf_tool: BaseTool):
    web_search_tool = FireCrawlWebSearchTool()
    retriever_agent = Agent(
        role="Data Retrieval Agent for Document and Web Sources",
        goal=(
            "For the given user query: {query}, extract all detailed and relevant information. "
            "If the query is related to the uploaded PDF document, exhaustively search it; otherwise, perform an in-depth web search."
        ),
        backstory="You are a highly skilled information retrieval expert.",
        verbose=True,
        tools=[pdf_tool, web_search_tool],
        llm=load_llm()
    )
    response_synthesizer_agent = Agent(
        role="Response Synthesizer and Detail Enhancer",
        goal=(
            "Synthesize the retrieved information into a comprehensive, detailed final response for the query: {query}."
        ),
        backstory="You excel in summarizing complex information clearly.",
        verbose=True,
        llm=load_llm()
    )
    retrieval_task = Task(
        description="Retrieve the most detailed information for the query: {query}.",
        expected_output="A detailed retrieval of all relevant data.",
        agent=retriever_agent
    )
    response_task = Task(
        description="Synthesize a final answer by integrating the retrieved data for the query: {query}.",
        expected_output="A comprehensive and coherent final answer.",
        agent=response_synthesizer_agent
    )
    crew = Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True
    )
    return crew

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_data" not in st.session_state:
    st.session_state.qa_data = []
if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None
if "crew" not in st.session_state:
    st.session_state.crew = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.qa_data = []
    gc.collect()

def display_pdf(file_bytes: bytes, file_name: str):
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
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

st.markdown("# Layla Chat Bot Version .2 (Document search + Webscrape fallback)")
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
            result_obj = st.session_state.crew.kickoff(inputs=inputs)
            final_answer = result_obj.raw
        for line in final_answer.split("\n"):
            full_response += line + "\n"
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.05)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    tool_output = st.session_state.pdf_tool._run(prompt)
    st.session_state.qa_data.append({
        "user_input": prompt,
        "retrieved_contexts": tool_output["chunks"],
        "response": full_response,
        "reference": ""
    })

st.markdown("---")
st.subheader("📊 RAGas Evaluation")
if st.button("Compute RAGas Metrics"):
    if not st.session_state.qa_data:
        st.warning("No Q–A pairs to evaluate.")
    else:
        evaluation_dataset = EvaluationDataset.from_list(st.session_state.qa_data)
        metrics = [faithfulness, answer_correctness, context_recall, context_precision]
        with st.spinner("Evaluating with RAGas..."):
            evaluation_result = evaluate(dataset=evaluation_dataset, metrics=metrics, llm=critic_llm, embeddings=openai_embeddings)
        df = pd.DataFrame(evaluation_result.scores)
        st.dataframe(df)
        st.bar_chart(df.transpose())
