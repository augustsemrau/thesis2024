"""Module containing the teaching agent system."""
from langchain_openai import ChatOpenAI




class TeachingAgent:
    """Class for the teaching agent system."""

    def __init__(self, llm: str="gpt-3.5-turbo-0125",
                ):
        """Initialize the Assessor Agent class."""
        self.model = ChatOpenAI(model_name=llm, temperature=0.5)

        return None




