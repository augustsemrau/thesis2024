"""Load vector store and create chatbot.

Load the persistent Chroma vector store and create a chatbot using OpenAI's API.
"""

import os
import openai
import streamlit as st

from thesis2024.datamodules.make_dataset import load_peristent_chroma_store
from thesis2024.models.agent import create_agent_chain

def get_llm_response(query):
    """Get response from OpenAI's API.

    :param query: Query to ask the chatbot.
    :return: Response from the chatbot.
    """
    ## Create the agent chain
    chain = create_agent_chain()

    ## Load persistent Chroma vector store
    chroma_instance = load_peristent_chroma_store(openai_embedding=True)

    ## Search the vector store for similar documents
    docs = chroma_instance.similarity_search(query)
    # docs = chroma_instance.max_marginal_relevance_search(query, k=3, fetch_k=5)

    ## Run the chain
    return chain.run(input_documents=docs, question=query)


if __name__ == '__main__':
    ## Retrieve OpenAI API key from environment variable
    openai.api_key  = os.environ['OPENAI_API_KEY']

    st.set_page_config(page_title="Doc Searcher", page_icon=":robot:")
    st.header("Query PDF Source")

    form_input = st.text_input('Enter Query')
    submit = st.button("Generate")

    if submit:
        st.write(get_llm_response(form_input))

