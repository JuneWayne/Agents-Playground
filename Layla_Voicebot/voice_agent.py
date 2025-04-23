import io
import os
import tempfile
import gc
import base64
import time
import uuid
import json

import streamlit as st
import openai
import requests
from dotenv import load_dotenv
from audiorecorder import audiorecorder
from io import BytesIO
from pydub import AudioSegment
from openai import OpenAI

from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from typing import List, Any
from pydantic import PrivateAttr
from markitdown import MarkItDown
from pinecone import Pinecone, ServerlessSpec

from evaluator import Evaluator

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv("../.env")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")

elevenlabs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
elevenlabs_voice_id = os.getenv("ELEVEN_LABS_VOICE_ID")

# ---------------------------
# Pinecone Setup
# ---------------------------
pc = Pinecone(api_key=pinecone_api_key)
serverless_spec = ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region)

# ---------------------------
# Audio Transcription Function
# ---------------------------
client = OpenAI()
def transcribe_audio(wav_bytes: bytes) -> str:
    audio_file = BytesIO(wav_bytes)
    audio_file.name = "audio.wav"
    audio_file.seek(0)
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""

# ---------------------------
# ElevenLabs Text-to-Speech Function
# ---------------------------
def elevenlabs_speak(text: str) -> BytesIO:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}"
    headers = {
        "xi-api-key": elevenlabs_api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error(f"Error from ElevenLabs TTS API: {response.text}")
        return None

# ---------------------------
# Real-Time Auto-Play Function
# ---------------------------
def play_audio_auto(audio_bytes: bytes):
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    audio_html = f"""
    <audio autoplay style="display:none;" 
           src="data:audio/mp3;base64,{audio_base64}">
    Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# ---------------------------
# Document Search Tool
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
        if self.index_name in pc.list_indexes().names():
            pc.delete_index(self.index_name)
            time.sleep(2)
        elif self.index_name not in pc.list_indexes().names():
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

    def _run(self, query: str) -> str:
        query_embedding = self._get_openai_embedding(query)
        response = self._index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        docs = [match["metadata"]["text"] for match in response["matches"]]
        return "\n___\n".join(docs)

# ---------------------------
# Web Search Tool
# ---------------------------
class FireCrawlWebSearchTool(BaseTool):
    name: str = "FireCrawlWebSearchTool"
    description: str = "A placeholder web search tool."

    def _run(self, query: str) -> str:
        return f"Web search result for query: {query}"

# ---------------------------
# Load LLM for Multi-Agent System
# ---------------------------
@st.cache_resource
def load_llm():
    return LLM(model="gpt-4o", verbose=True, temperature=1)

# ---------------------------
# Create Agents and Tasks
# ---------------------------
def create_agents_and_tasks(pdf_tool: Any = None):
    tools_list = []
    if pdf_tool is not None:
        tools_list.append(pdf_tool)
    tools_list.append(FireCrawlWebSearchTool())

    retriever_agent = Agent(
        role="Data Retriever",
        goal="Find all detailed, relevant information about {query}.",
        backstory="You're an expert researcher.",
        tools=tools_list,
        llm=load_llm(),
        verbose=True
    )

    responder_agent = Agent(
        role="Answer Synthesizer",
        goal="Craft a full, rich response using retrieved info about {query}.",
        backstory="You're an expert writer.",
        llm=load_llm(),
        verbose=True
    )

    return Crew(
        agents=[retriever_agent, responder_agent],
        tasks=[
            Task(
                description="Find the most complete information for the query: {query}",
                agent=retriever_agent,
                expected_output="A detailed extraction of relevant context and information regarding the query."
            ),
            Task(
                description="Write a clear, detailed response to the query: {query}",
                agent=responder_agent,
                expected_output="A comprehensive and well-articulated answer to the query."
            )
        ],
        process=Process.sequential,
        verbose=True
    )

# ---------------------------
# Streamlit Session State Setup
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None

if "crew" not in st.session_state:
    st.session_state.crew = None

if "evaluator" not in st.session_state:
    st.session_state.evaluator = Evaluator()

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def display_pdf(file_bytes: bytes, file_name: str):
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    st.markdown(f"### Preview of {file_name}")
    st.markdown(f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}"
        width="100%" height="600px" type="application/pdf"></iframe>
    """, unsafe_allow_html=True)

# ---------------------------
# Sidebar: PDF Upload & Indexing
# ---------------------------
with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    if uploaded_file and st.session_state.pdf_tool is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            with st.spinner("Indexing PDF..."):
                st.session_state.pdf_tool = DocumentSearchTool(file_path=tmp_path)
        st.success("PDF indexed!")
        display_pdf(uploaded_file.getvalue(), uploaded_file.name)

    st.button("Reset Chat", on_click=reset_chat)

# ---------------------------
# Main Interface: Chat Display
# ---------------------------
st.title("Comprehensive Chatbot")
st.write("RAG + Webscrape-Fallback + Sequential Multi-agent + Audio Transcription & Output + Self-Evaluation ")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# Voice Input Section
# ---------------------------
st.subheader("Voice Input")
audio = audiorecorder("", "")

if len(audio) > 0:
    st.audio(audio.export().read())
    st.write(
        f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, "
        f"Duration: {audio.duration_seconds} seconds"
    )

    audio_buffer = io.BytesIO()
    audio.export(audio_buffer, format="wav", parameters=["-ar", "16000"])
    wav_bytes = audio_buffer.getvalue()

    with st.spinner("Transcribing..."):
        prompt = transcribe_audio(wav_bytes)

    if prompt.strip():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**You said:** {prompt}")

        if st.session_state.crew is None:
            st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)

        with st.chat_message("assistant"):
            response_box = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                crew_output = st.session_state.crew.kickoff(inputs={"query": prompt})
                answer  = crew_output.raw
                context = crew_output.tasks_output[0].raw

            for i, line in enumerate(answer.split("\n")):
                full_response += line + ("\n" if i < len(answer.split("\n")) - 1 else "")
                response_box.markdown(full_response + "▌")
                time.sleep(0.05)
            response_box.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# Evaluation for Audio Input
# -----------------------------

        with st.spinner("Evaluation in progress..."):
            
            with st.expander("Evaluation"):
                try:
                    df = st.session_state.evaluator.evaluate_responses(
                        questions=prompt,
                        answer=answer,
                        references=context,
                    )
                    st.dataframe(df)

                    metric_cols = [
                        "answer_correctness",
                        "answer_relevancy",
                        "context_recall",
                        "context_precision",
                        "faithfulness"
                    ]

                    def flatten_to_scalar(x):
                        if isinstance(x, (list, tuple, np.ndarray)):
                            return x[0] if len(x) > 0 else np.nan
                        return x

                    df[metric_cols] = df[metric_cols].applymap(flatten_to_scalar)

                    df[metric_cols] = df[metric_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
                    df[metric_cols] = df[metric_cols].fillna(0.0)

                    scores = df.loc[0, metric_cols].values.astype(float)

                    cmap = cm.get_cmap("RdYlGn")
                    norm = plt.Normalize(0.0, 1.0)
                    colors = cmap(norm(scores))
                    
                    metric_cols_renamed = [
                        'Factual Correctness',
                        'Relevance to Query',
                        'Context Adequacy',
                        'Context Usefulness',
                        'Answer Relevancy to Context'
                    ]

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.barh(metric_cols, scores, color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Score (0–1)")
                    ax.set_title("Evaluation Metrics")
                    ax.set_yticklabels(metric_cols_renamed)
                    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
                    for label in ax.get_yticklabels():
                        label.set_position((-0.05, label.get_position()[1]))

                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.15, fraction=0.05)
                    cbar.set_label("Response Quality")
                    cbar.ax.text(1.2, -0.13, "Low quality\nresponse",  va="bottom", ha="left")
                    cbar.ax.text(1.2, 1.13, "High quality\nresponse", va="top",    ha="left")

                    avg = float(scores.mean())
                    cbar.ax.hlines(y=avg, xmin=0, xmax=1, color="black", linewidth=2)

                    st.pyplot(fig)

                    N = len(metric_cols)
                    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                    angles += angles[:1]

                    values = scores.tolist()
                    values += values[:1]

                    fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                    ax2.set_xticks(angles[:-1])
                    ax2.set_xticklabels(metric_cols_renamed, fontweight='bold')
                    ax2.set_rlabel_position(90)
                    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
                    ax2.set_yticklabels(['0.2','0.4','0.6','0.8'], color="grey", size=8)
                    ax2.set_ylim(0, 1)

                    ax2.plot(angles, values, linewidth=2, linestyle='solid', label="Metrics")
                    ax2.fill(angles, values, alpha=0.25)
                    ax2.set_title("Metric Spider Chart", y=1.1)
                    ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

                    st.pyplot(fig2)

                except Exception as e:
                    st.error(f"Something's wrong in evaluation: {e}")
        if avg <= 0.5:
            st.warning("The response quality is low. Please try rephrasing your question or providing more context.")
        elif avg <= 0.7:
            st.info("The response quality is moderate. Consider refining your question for better results.")
        else:
            st.success("The response quality is good. You can proceed with the information provided.")

        audio_response = elevenlabs_speak(answer)
        if audio_response:
            play_audio_auto(audio_response.getvalue())

# ---------------------------
# Text Input Section
# ---------------------------
prompt = st.chat_input("Or type a question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.crew is None:
        st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)

    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            crew_output = st.session_state.crew.kickoff(inputs={"query": prompt})
            answer  = crew_output.raw
            context = crew_output.tasks_output[0].raw

        for i, line in enumerate(answer.split("\n")):
            full_response += line + ("\n" if i < len(answer.split("\n")) - 1 else "")
            response_box.markdown(full_response + "▌")
            time.sleep(0.15)
        response_box.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------------------
# Evaluation for text input
# ---------------------------

    with st.spinner("Evaluation in progress..."):
        with st.expander("Evaluation"):
            try:
                df = st.session_state.evaluator.evaluate_responses(
                    questions=prompt,
                    answer=answer,
                    references=context,
                )
                st.dataframe(df)

                metric_cols = [
                    "answer_correctness",
                    "answer_relevancy",
                    "context_recall",
                    "context_precision",
                    "faithfulness"
                ]

                def flatten_to_scalar(x):
                    if isinstance(x, (list, tuple, np.ndarray)):
                        return x[0] if len(x) > 0 else np.nan
                    return x

                df[metric_cols] = df[metric_cols].applymap(flatten_to_scalar)

                df[metric_cols] = df[metric_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
                df[metric_cols] = df[metric_cols].fillna(0.0)

                scores = df.loc[0, metric_cols].values.astype(float)

                cmap = cm.get_cmap("RdYlGn")
                norm = plt.Normalize(0.0, 1.0)
                colors = cmap(norm(scores))
                
                metric_cols_renamed = [
                    'Factual Correctness',
                    'Relevance to Query',
                    'Context Adequacy',
                    'Context Usefulness',
                    'Answer Relevancy to Context'
                ]

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(metric_cols, scores, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Score (0–1)")
                ax.set_title("Evaluation Metrics")
                ax.set_yticklabels(metric_cols_renamed)
                plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
                for label in ax.get_yticklabels():
                    label.set_position((-0.05, label.get_position()[1]))

                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.15, fraction=0.05)
                cbar.set_label("Response Quality")
                cbar.ax.text(1.2, -0.13, "Low quality\nresponse",  va="bottom", ha="left")
                cbar.ax.text(1.2, 1.13, "High quality\nresponse", va="top",    ha="left")

                avg = float(scores.mean())
                cbar.ax.hlines(y=avg, xmin=0, xmax=1, color="black", linewidth=2)

                st.pyplot(fig)

                N = len(metric_cols)
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                angles += angles[:1]

                values = scores.tolist()
                values += values[:1]

                fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                ax2.set_xticks(angles[:-1])
                ax2.set_xticklabels(metric_cols_renamed, fontweight='bold')
                ax2.set_rlabel_position(90)
                ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
                ax2.set_yticklabels(['0.2','0.4','0.6','0.8'], color="grey", size=8)
                ax2.set_ylim(0, 1)

                ax2.plot(angles, values, linewidth=2, linestyle='solid', label="Metrics")
                ax2.fill(angles, values, alpha=0.25)
                ax2.set_title("Metric Spider Chart", y=1.1)
                ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Something's wrong in evaluation: {e}")
    if avg <= 0.5:
        st.warning("The response quality is low. Please try rephrasing your question or providing more context.")
    elif avg <= 0.7:
        st.info("The response quality is moderate. Consider refining your question for better results.")
    else:
        st.success("The response quality is good. You can proceed with the information provided.")
