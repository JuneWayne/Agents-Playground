import os
import warnings
import streamlit as st
import time
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

warnings.filterwarnings("ignore")
load_dotenv()

# Define directories for ChromaDB storage
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
os.makedirs(db_dir, exist_ok=True)


def create_vector_store(url, persistent_directory):
    """Crawl the user-input website, split content, create embeddings, and persist in ChromaDB."""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        st.error("❌ FIRECRAWL_API_KEY is missing. Please set it in your environment variables.")
        return

    st.info(f"🔍 Crawling website: {url} ...")
    loader = FireCrawlLoader(api_key=api_key, url=url, mode="scrape")
    docs = loader.load()
    st.success("✅ Website crawled successfully!")

    # Convert metadata values to strings if they are lists
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # Step 2: Split the crawled content
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    st.info(f"📄 Processed {len(split_docs)} document chunks.")

    # Step 3: Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step 4: Create and persist ChromaDB
    st.info("🗂️ Storing embeddings in ChromaDB...")
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
    st.success("✅ Vector store created successfully!")


# Streamlit UI
def main():
    st.title("🌐 Web Scraper & AI-Powered Search")
    st.subheader("Scrape a website, store its content, and search through it!")

    # User inputs a website URL
    url = st.text_input("🔗 Enter a website URL to scrape:", placeholder="e.g., https://apple.com")

    if st.button("Scrape Website"):
        if url.strip():
            # Create a directory for this specific website
            website_name = url.replace("https://", "").replace("http://", "").replace("/", "_")
            persistent_directory = os.path.join(db_dir, f"chroma_db_{website_name}")
            os.makedirs(persistent_directory, exist_ok=True)

            st.session_state["persistent_directory"] = persistent_directory
            create_vector_store(url, persistent_directory)
        else:
            st.warning("⚠️ Please enter a valid website URL.")

    # Load vector store (after scraping)
    if "persistent_directory" in st.session_state and os.path.exists(st.session_state["persistent_directory"]):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=st.session_state["persistent_directory"], embedding_function=embeddings)

        # Query input
        query = st.text_input("🔍 Enter your search query:", placeholder="e.g., What is Apple Intelligence?")

        if st.button("Search"):
            if query.strip():
                st.balloons()
                loading_placeholder = st.empty()
                loading_placeholder.image("Data_file/loading_cat.gif", caption="Searching...")

                retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(query)

                loading_placeholder.empty()

                if relevant_docs:
                    st.success("✅ Here are the most relevant results:")
                    for i, doc in enumerate(relevant_docs, 1):
                        st.markdown(f"**Document {i}:**\n{doc.page_content}\n")
                        if doc.metadata:
                            st.markdown(f"📌 **Source:** {doc.metadata.get('source', 'Unknown')}\n")
                else:
                    st.warning("⚠️ No relevant results found. Try a different query!")
            else:
                st.warning("⚠️ Please enter a search query.")


if __name__ == "__main__":
    main()
