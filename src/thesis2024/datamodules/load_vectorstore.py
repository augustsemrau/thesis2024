"""Initiates vectorstore and return a retriever for use in RAG setups."""

import os
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings



def load_peristent_chroma_store(openai_embedding):
    """Load a persistent Chroma vector store.

    :param persist_path: Path to the persistent Chroma vector store.
    :return: Chroma vector store.
    """
    ## Initiate embedding function
    if openai_embedding:
        embedding_func = OpenAIEmbeddings()
    else:
        embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_dir = os.path.join(os.getcwd(), 'data/processed/chroma')
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_func)


