"""Load vector store and create chatbot.

Load the persistent Chroma vector store and create a chatbot using OpenAI's API.
"""

import os
import openai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

def load_peristent_chroma_store(embedding_function,
                                persist_dir='data/processed/chroma'):
    """Load a persistent Chroma vector store.

    :param persist_path: Path to the persistent Chroma vector store.
    :return: Chroma vector store.
    """
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_function)



if __name__ == '__main__':

    ## Retrieve OpenAI API key from environment variable
    openai.api_key  = os.environ['OPENAI_API_KEY']

    ## Initiate embedding function
    # embedding_func = OpenAIEmbeddings()
    embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_path = os.path.join(os.getcwd(), 'data/processed/chroma')

    ## Load persistent Chroma vector store
    chroma_instance = load_peristent_chroma_store(embedding_function=embedding_func, persist_dir=persist_path)

    ## Query the Chroma vector store
    query = "Piss off ghost!"
    # query = 'So let me show you a video now. Load the big screen, please. So Ill show you a video now that was from Dean Pomerleau at some work he did at Carnegie Mellon on applied supervised learning to get a car to drive itself. This is work on a vehicle known as Alvin.  It was done sort of about 15 years ago, and I think it was a very elegant example of the sorts of things you can get supervised or any algorithms to do.'
    search_results = chroma_instance.similarity_search(query, k=3)
    # search_results = chroma_instance.max_marginal_relevance_search(query, k=3, fetch_k=5)



