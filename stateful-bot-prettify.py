import os
import warnings
import streamlit as st
import time
from dotenv import load_dotenv

# Firecrawl / LangChain / Pinecone imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import FireCrawlLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

import pinecone
from pinecone import ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeVectorStore

warnings.filterwarnings("ignore")
load_dotenv()

def create_vector_store(url, index_name):
    """Crawl the user-input website, split content, create embeddings, and store in Pinecone."""
    # 1) Firecrawl API Key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        st.error("❌ FIRECRAWL_API_KEY is missing.")
        return

    # 2) Crawl the site
    st.info(f"🔍 Crawling {url} ...")
    loader = FireCrawlLoader(api_key=api_key, url=url, mode="scrape")
    docs = loader.load()
    st.success("✅ Website crawled successfully!")

    # 3) Convert list metadata to string
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # 4) Split the documents
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    st.info(f"Processed {len(split_docs)} document chunks.")

    # 5) Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 6) Initialize Pinecone client (no pinecone.init(...) anymore!)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    project_name = os.getenv("PINECONE_PROJECT_NAME")  # if needed

    if not pinecone_api_key:
        st.error("❌ PINECONE_API_KEY is missing.")
        return

    # Create a Pinecone instance
    pc = pinecone.Pinecone(api_key=pinecone_api_key, project_name=project_name)
    # If you want a specific region, do:
    # pc.create_index(... spec=ServerlessSpec(cloud='aws', region='us-west-2'))

    # 7) Create the index if it doesn’t exist
    existing_indexes = pc.list_indexes().names()  # returns a list of index names
    if index_name not in existing_indexes:
        st.info(f"Creating Pinecone index '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=1536,       # for text-embedding-ada-002 or "text-embedding-3-small"
            metric="cosine",
            # spec=ServerlessSpec(cloud='aws', region='us-west-2') # optional
        )
        st.info(f"Index '{index_name}' created!")

    # 8) Store documents in Pinecone
    st.info("🚀 Storing docs in Pinecone index...")
    PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=index_name,
        pinecone_client=pc
    )
    st.success("✅ Documents successfully stored in Pinecone!")


def create_question_generator(llm):
    """Chain to rephrase the question based on chat history."""
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
    """Chain that summarizes the docs to answer user questions."""
    summary_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful AI assistant. Summarize the following context and answer the user's question.

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


def create_conversational_chain(llm, index_name):
    """
    Creates a conversational chain:
    1) Rephrase user queries
    2) Summarize docs
    3) Retrieve from Pinecone
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    project_name = os.getenv("PINECONE_PROJECT_NAME")
    pc = pinecone.Pinecone(api_key=pinecone_api_key, project_name=project_name)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1) Load existing index from Pinecone
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        pinecone_client=pc
    )

    # 2) Build retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 3) Build question generator + summary chain
    question_generator_chain = create_question_generator(llm)
    summary_chain = create_summary_chain(llm)

    # 4) Create the final conversational chain
    conv_chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator_chain,
        combine_docs_chain=summary_chain,
    )
    return conv_chain


def main():
    st.title("Layla Web-Scraping Chat Bot (Fixed Pinecone Version)")
    st.subheader("Scrape a website, store embeddings in Pinecone, then ask questions!")

    # Let user input a website URL
    url = st.text_input("Website URL:", placeholder="https://example.com")

    # Derive an index name from the URL
    if url.strip():
        index_name = "layla_" + url.replace("https://", "").replace("http://", "").replace("/", "_").lower()
    else:
        index_name = "layla_default"

    # Scrape and store button
    if st.button("Scrape Website"):
        if url.strip():
            create_vector_store(url, index_name)
            st.session_state["pinecone_index"] = index_name
        else:
            st.warning("⚠ Please enter a valid website URL.")

    # If we have an index
    if "pinecone_index" not in st.session_state and index_name:
        st.session_state["pinecone_index"] = index_name

    if "pinecone_index" in st.session_state:
        # Create the chat model
        chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-4o", verbose=True)

        # Create the conversation chain
        chat_chain = create_conversational_chain(chat_model, st.session_state["pinecone_index"])

        # Keep track of user Q & A
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("Ask a question about the website:", placeholder="What do we have here?")

        if st.button("Ask Layla"):
            if query.strip():
                st.balloons()
                loader_place = st.empty()
                loader_place.image("Data_file/loading_cat.gif", caption="Thinking...")

                response = chat_chain({
                    "question": query,
                    "chat_history": st.session_state.chat_history
                })

                loader_place.empty()

                if response.get("answer"):
                    st.success("Layla says:")
                    st.write(response["answer"])
                    # Save the conversation
                    st.session_state.chat_history.append((query, response["answer"]))
                else:
                    st.warning("No relevant answer found. Try another question.")
            else:
                st.warning("⚠ Please enter a question.")

        # Show chat history
        if st.session_state.chat_history:
            st.write("### Chat History:")
            for i, (user_q, bot_a) in enumerate(st.session_state.chat_history, start=1):
                st.write(f"**Q {i}**: {user_q}")
                st.write(f"**A {i}**: {bot_a}")


if __name__ == "__main__":
    main()
