"""Load vector store and create chatbot.

Load the persistent Chroma vector store and create a chatbot using OpenAI's API.
"""

import os
import openai

from thesis2024.datamodules.make_dataset import load_peristent_chroma_store






if __name__ == '__main__':

    ## Retrieve OpenAI API key from environment variable
    openai.api_key  = os.environ['OPENAI_API_KEY']




    ## Load persistent Chroma vector store
    chroma_instance = load_peristent_chroma_store(openai_embedding=True)

    ## Query the Chroma vector store
    query = "I like AI!"
    # query = 'So let me show you a video now. Load the big screen, please. So Ill show you a video now that was from Dean Pomerleau at some work he did at Carnegie Mellon on applied supervised learning to get a car to drive itself. This is work on a vehicle known as Alvin.  It was done sort of about 15 years ago, and I think it was a very elegant example of the sorts of things you can get supervised or any algorithms to do.'
    search_results = chroma_instance.similarity_search(query, k=3)
    # search_results = chroma_instance.max_marginal_relevance_search(query, k=3, fetch_k=5)
    print(search_results)


