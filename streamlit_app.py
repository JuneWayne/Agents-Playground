import os
import warnings
import streamlit as st
from dotenv import load_dotenv

# LangChain & related imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ConversationalRetrievalChain, LLMChain

warnings.filterwarnings("ignore")
load_dotenv()

def create_question_generator(llm):
    question_generator_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow-up question,
rephrase the follow-up question to be a standalone question.

Conversation:
{chat_history}
Follow-up question: {question}

Standalone question:"""
    )
    return LLMChain(llm=llm, prompt=question_generator_prompt)

def create_stuff_chain(llm):
    
    custom_template = """
You are a helpful assistant. Use the following context to answer the user's question, remember you may access multiple files at the same time,
you can be lengthy in your answer, but you have to produce references to which specific document you are referring to each time you generate an opinion

Context:
{context}

Question:
{question}

Answer:
"""
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_template
    )

    return StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=custom_prompt),
        document_variable_name="context"
    )

def main():
    st.title("Meeting/Interview Notes Q&A (Direct Answers)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    # Create your chat model
    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o")

    # Build question generator (if your version requires it)
    question_generator_chain = create_question_generator(llm=chat)

    # Create “stuff” chain with simpler prompt
    stuff_chain = create_stuff_chain(llm=chat)

    # Build ConversationalRetrievalChain
    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=stuff_chain,
        question_generator=question_generator_chain
    )

    query = st.text_input("Enter your question:")
    if st.button("Submit") and query.strip():
        res = qa({"question": query, "chat_history": st.session_state.chat_history})

        st.write("### Response:")
        st.write(res["answer"])

        st.session_state.chat_history.append((query, res["answer"]))

    if st.session_state.chat_history:
        st.write("### Chat History:")
        for i, (user_q, bot_a) in enumerate(st.session_state.chat_history, start=1):
            st.write(f"**Q {i}:** {user_q}")
            st.write(f"**A {i}:** {bot_a}")

if __name__ == "__main__":
    main()
