import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

warnings.filterwarnings("ignore")
load_dotenv()

chat_history = []

def print_conversation(query, response, chat_history):
    print("\n" + "="*50)
    print("QUERY:")
    print("-"*50)
    print(query)
    
    print("\nRESULTS:")
    print("-"*50)
    if isinstance(response, dict):

        answer = response.get("answer", "")
        print(answer)
        if "source_documents" in response:
            print("\nSOURCE DOCUMENTS:")
            print("-"*50)
            for i, doc in enumerate(response["source_documents"], 1):
                source = doc.metadata.get("source", "unknown")
                print(f"{i}. {source}")
    else:
        print(response)
    
    if chat_history:
        print("\nCHAT HISTORY:")
        print("-"*50)
        for i, (q, a) in enumerate(chat_history, 1):
            print(f"{i}. Query: {q}")
            print(f"   Answer: {a}")
    print("="*50 + "\n")

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], 
        embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever()
    )  

    query1 = "What did Jonathan say about MLEs? Give me 5 points"
    res1 = qa({"question": query1, "chat_history": chat_history})
    print_conversation(query1, res1, chat_history)
    chat_history.append((res1["question"], res1["answer"]))

    query2 = "Can you give me an additional two points?"
    res2 = qa({"question": query2, "chat_history": chat_history})
    chat_history.append((res2["question"], res2["answer"]))
    print_conversation(query2, res2, chat_history)
