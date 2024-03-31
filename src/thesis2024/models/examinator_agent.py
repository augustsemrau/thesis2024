"""Functions for the agent model."""

# LangChain imports
from langchain import hub
from langchain_core.output_parsers import StrOutputParser


from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union

from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from typing_extensions import TypedDict

# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str








class ExaminatorAgent:
    """Multi-Agent coding LangGraph model."""

    def __init__(self, llm: str="gpt-3.5-turbo-0125"):
        """Initialize the MultiAgent class."""
        self.llm = llm
        return None

    # Helper functions
    def create_agent(self, tools, system_message):
        """Create arbitrary agent."""
        functions = [format_tool_to_openai_function(t) for t in tools]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        return prompt | self.llm.bind_functions(functions)




if __name__ == "__main__":

    coding_class = CodingMultiAgent(llm="gpt-3.5-turbo-0125")

    coding_graph = coding_class.instanciate_graph()
    print(coding_graph)










