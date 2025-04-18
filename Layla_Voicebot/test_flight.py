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
    """
    Sends the provided WAV bytes to OpenAI's transcription endpoint.
    """
    audio_file = BytesIO(wav_bytes)
    audio_file.name = "audio.wav"  # Helps the API detect the file type
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
    """
    Uses the ElevenLabs API to convert text to human-sounding speech.
    Returns a BytesIO stream of the MP3 audio.
    """
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
    """
    Injects an HTML <audio> tag with autoplay (and hidden controls)
    to automatically play the provided MP3 audio.
    """
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    audio_html = f"""
    <audio autoplay style="display:none;" src="data:audio/mp3;base64,{audio_base64}">
    Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# ---------------------------
# Document Search Tool (PDF indexing via Pinecone & MarkItDown)
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

    def _run(self, query: str) -> str:
        query_embedding = self._get_openai_embedding(query)
        response = self._index.query(vector=query_embedding, top_k=5, include_metadata=True)
        docs = [match["metadata"]["text"] for match in response["matches"]]
        return "\n___\n".join(docs)

# ---------------------------
# Placeholder Web Search Tool
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
# Create Agents and Tasks (Multi-Agent System)
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
st.title("Layla Voicebot (Transcription Only Mode)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# Voice Input Section (Using Reference UI)
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
        user_voice_text = transcribe_audio(wav_bytes)

    if user_voice_text.strip():
        st.session_state.messages.append({"role": "user", "content": user_voice_text})
        with st.chat_message("user"):
            st.markdown(f"**You said:** {user_voice_text}")

        if st.session_state.crew is None:
            st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)

        with st.chat_message("assistant"):
            response_box = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                result = st.session_state.crew.kickoff(inputs={"query": user_voice_text}).raw
 
            for i, line in enumerate(result.split("\n")):
                full_response += line + ("\n" if i < len(result.split("\n")) - 1 else "")
                response_box.markdown(full_response + "▌")
                time.sleep(0.05)
            response_box.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": result})

        audio_response = elevenlabs_speak(result)
        if audio_response:
            play_audio_auto(audio_response.getvalue())

# ---------------------------
# Text Input Section (Alternative Chat Interface)
# ---------------------------
prompt = st.chat_input("Or type a question...")
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
            result = st.session_state.crew.kickoff(inputs={"query": prompt}).raw
        for i, line in enumerate(result.split("\n")):
            full_response += line + ("\n" if i < len(result.split("\n")) - 1 else "")
            response_box.markdown(full_response + "▌")
            time.sleep(0.15)
        response_box.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": result})
    audio_response = elevenlabs_speak(result)
    if audio_response:
        play_audio_auto(audio_response.getvalue())
