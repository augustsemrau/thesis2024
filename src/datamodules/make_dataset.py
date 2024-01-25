"""Create the dataset for the thesis.

Making the dataset for the thesis, by loading datasources (currently PDFs)
and creating a persistent Chroma vector store from them.
"""


import os
import openai
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def create_persistent_chroma_store(embedding_func,
                                   directory='docs/',
                                   persist_path='chroma/',
                                   chunk_size=1000,
                                   chunk_overlap=150):
    """Create Vector Store.

    Create a persistent Chroma vector store from PDF documents in a directory.
    :param directory: Path to the directory containing PDF documents.
    :param store_path: Path to save the persistent Chroma vector store.
    :param chunk_size: Size of text chunks for each document.
    :param chunk_overlap: Overlap between consecutive text chunks.
    :return: None
    """
    # Process each PDF file in the directory
    all_docs = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)

            # Load PDF document
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
            )
            docs = text_splitter.split_documents(documents)
            print(f"{len(docs)} chunks created from {filename}")
            all_docs.extend(docs)

    # Save the Chroma vector store persistently
    # Chroma.from_documents(all_docs, embedding_function=OpenAIEmbeddings(), persist_directory=persist_path)
    Chroma.from_documents(all_docs, embedding_func, persist_directory=persist_path)
    print(f"Chroma vector store created and saved at {persist_path}")

    return


if __name__ == '__main__':

    ## Retrieve OpenAI API key from environment variable
    openai.api_key  = os.environ['OPENAI_API_KEY']

    ## Initiate embedding function
    embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Get the data and process it
    pass
