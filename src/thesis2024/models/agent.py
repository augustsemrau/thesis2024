"""Functions for the agent model."""


from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

def create_agent_chain():
    """Create agent chain.

    Create a chain for the agent model.
    :return: Chain.
    """
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

