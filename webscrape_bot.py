import os
import warnings
import streamlit as st
import time
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

warnings.filterwarnings("ignore")
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
os.makedirs(db_dir, exist_ok=True)

def create_vector_store(url, persistent_directory):
    """Crawl the user-input website, split content, create embeddings, and persist in ChromaDB."""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        st.error("FIRECRAWL_API_KEY is missing. Please set it in your environment variables.")
        return

    st.info(f"🔍 Crawling: {url} furiously...")
    loader = FireCrawlLoader(api_key=api_key, url=url, mode="scrape")
    docs = loader.load()
    st.success("Website crawled successfully!")

    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    st.info(f"Processed {len(split_docs)} document chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    st.info("Storing embeddings in ChromaDB...")
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
    st.success("Vector store created successfully!")

def create_question_generator(llm):
    question_gen_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=(
            "Given the following conversation and a follow-up question, "
            "rephrase the follow-up question to be a standalone question.\n\n"
            "Conversation:\n{chat_history}\n\n"
            "Follow-up question: {question}\n\n"
            "Standalone question:"
        ),
    )
    return LLMChain(llm=llm, prompt=question_gen_prompt)

def create_summary_chain(llm):
    summary_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful AI assistant that presents and summarizes all of the information concisely to the best of your ability.
Given the retrieved web content below, generate a well-rounded, well-structured answer to the user's question.
After generating a response, you must also provide relevant hyperlinks to the user as well.

Context:
{context}

User Question:
{question}

Answer:
"""
    )

    return StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=summary_prompt),
        document_variable_name="context"
    )

def create_conversational_chain(llm, retriever):
    """Creates a chatbot-like conversational retrieval chain with question rephrasing and summarization."""
    question_generator_chain = create_question_generator(llm)

    summary_chain = create_summary_chain(llm)

    conv_chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator_chain,
        combine_docs_chain=summary_chain
    )
    return conv_chain

def main():
    st.title("Layla Web-Scraping Chat Bot 🐈")
    st.subheader("Ask Layla to scrape any websites for you!")

    url = st.text_input("Enter a website URL to scrape:", placeholder="e.g., https://LaylaCat.com")

    if st.button("Scrape Website"):
        if url.strip():
            website_name = url.replace("https://", "").replace("http://", "").replace("/", "_")
            persistent_directory = os.path.join(db_dir, f"chroma_db_{website_name}")
            os.makedirs(persistent_directory, exist_ok=True)

            st.session_state["persistent_directory"] = persistent_directory
            create_vector_store(url, persistent_directory)
        else:
            st.warning("⚠️ Please enter a valid website URL.")

    if "persistent_directory" in st.session_state and os.path.exists(st.session_state["persistent_directory"]):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=st.session_state["persistent_directory"], embedding_function=embeddings)

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-4o", verbose=True)

        chat_chain = create_conversational_chain(llm=chat_model, retriever=retriever)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("💬 Ask a question about the website:", placeholder="e.g., What is Apple Intelligence?")

        if st.button("Ask AI"):
            if query.strip():
                st.balloons()
                loading_placeholder = st.empty()
                loading_placeholder.image("Data_file/loading_cat.gif", caption="Thinking...")

                response = chat_chain({
                    "question": query,
                    "chat_history": st.session_state.chat_history
                })

                loading_placeholder.empty()

                if response.get("answer"):
                    st.success("🧠 AI Response:")
                    st.markdown(response["answer"])

                    st.session_state.chat_history.append((query, response["answer"]))
                else:
                    st.warning("⚠️ No relevant answer found. Try a different question.")
            else:
                st.warning("⚠️ Please enter your question.")

        if st.session_state.chat_history:
            st.write("### 🔄 Chat History:")
            for i, (user_q, bot_a) in enumerate(st.session_state.chat_history, start=1):
                st.write(f"**Q {i}:** {user_q}")
                st.write(f"**A {i}:** {bot_a}")

if __name__ == "__main__":
    main()
