from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4")

def createRAGDB():
    loader = TextLoader("document.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)

    db.save_local("RAG-DB")

    print("FAISS index created and saved locally to 'my_faiss_index'.")

def answerWithRAG():
    db = FAISS.load_local("RAG-DB", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return retriever, rag_chain

# if __name__ == "__main__": # Example of how to run it
#     query = "What notable executive orders did Obama pass in his second term?"
#     retriever, chain = answerWithRAG()
#
#     print("---" * 20)
#     print(f"Retrieving chunks for query: '{query}'")
#     retrieved_docs = retriever.invoke(query)  # Or use retriever.get_relevant_documents(query)
#
#     # Print the content of each retrieved chunk
#     for i, doc in enumerate(retrieved_docs):
#         print(f"--- Chunk {i + 1} ---")
#         print(doc.page_content)
#         # You can also print metadata if you have it
#         # print(f"Source: {doc.metadata.get('source', 'N/A')}")
#     print("---" * 20)
#     # --- End of added part ---
#
#     # Now you can continue to run the full chain as before
#     # This will use the same retrieval process internally
#     response = chain.invoke(query)
#     print("\n--- Final RAG Answer ---")
#     print(response)
#
#
#     # response = chain.invoke(query)
#     # print(response)
#
#     # Example: Ask another question
#     # query2 = "Another question about the document"
#     # response2 = rag_chain.invoke(query2)
#     # print(response2)