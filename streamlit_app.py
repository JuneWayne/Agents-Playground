import os
import warnings
import streamlit as st
import time
from dotenv import load_dotenv

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
You are a helpful assistant specialized in analyzing meeting/interview notes.
Given the context (documents) below, summarize the key information, highlight any
action items, and then answer the user's question.

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
    st.title("Hello! My name is Layla, I am your interviewing assistant!")
    #chat_history = [("Assistant", "Hello! My name is Layla, I am your interviewing assistant, I have your notes in file already, how may I help you?")]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=1, model_name="gpt-4o")

    question_generator_chain = create_question_generator(llm=chat)

    stuff_chain = create_stuff_chain(llm=chat)

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=stuff_chain,
        question_generator=question_generator_chain
    )
    
    st.subheader('I have all your documents in file, how may I help you?')
    query = st.text_input("# i.e. Summarize Joe's interview notes for me in bullet points ")
    if st.button("Submit") and query.strip():
        st.balloons()
        loading_placeholder = st.empty()
        loading_placeholder.image("Data_file/loading_cat.gif", caption="loading...")
    
        res = qa({"question": query, "chat_history": st.session_state.chat_history})
        loading_placeholder.empty()
        #st.subheader('Thinking...')
        #st.progress(10)
        #with st.spinner('Looking over the corresponding documents...'):
            #time.sleep(10)
        with st.success("Ah! There you go!"):
            time.sleep(1)
        st.write("### Here's what I got:")
        st.write(res["answer"])
        st.write("### Let me know if you'd like me to find something different!")

        st.session_state.chat_history.append((query, res["answer"]))

    if st.session_state.chat_history:
        st.write("### Chat History:")
        for i, (user_q, bot_a) in enumerate(st.session_state.chat_history, start=1):
            st.write(f"**Q {i}:** {user_q}")
            st.write(f"**A {i}:** {bot_a}")

if __name__ == "__main__":
    main()
