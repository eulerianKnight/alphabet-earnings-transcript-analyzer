import os
from dotenv import load_dotenv, find_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv(find_dotenv())

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT_REGION']
)

def ingest_pdf() -> None:
    # Instantiate Document loader
    loader = PyPDFLoader(file_path='alphabet_earnings_release/2023Q1.pdf')
    # Load the document
    raw_document = loader.load()
    print(f"Loaded {len(raw_document) } documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        separators=['/n/n', '/n', " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_document)
    print(f"Splitted into {len(documents)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    Pinecone.from_documents(
        documents=documents, 
        embedding=embeddings, 
        index_name='alphabet-earnings-transcript'
    )
    print("ADDED TO PINECONE STORE!")

if __name__ == "__main__":
    ingest_pdf()