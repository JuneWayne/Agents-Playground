import os
import warnings
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ConversationalRetrievalChain, LLMChain

warnings.filterwarnings("ignore")
load_dotenv()

def print_conversation(query, response, chat_history):
    print("\n" + "="*50)
    print("QUERY:")
    print(query)
    print("\nRESULT:")
    print(response["answer"])
    print("\nCHAT HISTORY:")
    for i, (q, a) in enumerate(chat_history, start=1):
        print(f"{i}. Q: {q}")
        print(f"   A: {a}")
    print("="*50 + "\n")

def main():
    chat_history = []

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )


    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o")
    question_generator_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow-up question,
rephrase the follow-up question to be a standalone question.

Conversation:
{chat_history}
Follow-up question: {question}

Standalone question:"""
    )
    question_generator_chain = LLMChain(llm=chat, prompt=question_generator_prompt)

    custom_template = """
You are a helpful assistant specialized in analyzing meeting/interview notes.
Given the context (documents) below, summarize the key information, highlight any
action items, and then answer the user's question.

Context:
{context}

Question:
{question}

Provide your answer below:
"""
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_template
    )

    stuff_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=chat, prompt=custom_prompt),
        document_variable_name="context"
    )

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=stuff_chain,
        question_generator=question_generator_chain
    )

    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower().strip() == "exit":
            break

        res = qa({"question": query, "chat_history": chat_history})

        print_conversation(query, res, chat_history)


        chat_history.append((query, res["answer"]))

if __name__ == "__main__":
    main()
