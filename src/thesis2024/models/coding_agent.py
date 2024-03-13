"""Functions for the agent model."""

# LangChain imports
from langchain import hub
from langchain_core.output_parsers import StrOutputParser




# Local imports
from thesis2024.utils import GraphState





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




class SimpleAgeneicAgent:
    """SimpleAgeneicAgent class.

    A simple agent that uses the LangChain library to answer questions.
    """

    def __init__(self,
                generate_model: str="gpt-3.5-turbo"):
        """Initialize the SimpleAgeneicAgent class."""
        self.generate_model = generate_model
        return None


    def generate(self, state):
        """Node Function: Generate answer.

        Args:
        ----
            state (dict): The current graph state

        Returns:
        -------
            state (dict): New key added to state, generation, that contains LLM generation

        """
        print("---GENERATE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")


        # LLM
        llm = ChatOpenAI(model_name=self.generate_model, temperature=0, streaming=True)

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {
            "keys": {"documents": documents, "question": question, "generation": generation}
        }
















