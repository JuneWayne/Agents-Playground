import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

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

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
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
