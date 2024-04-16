"""Module containing the teaching agent system."""

import os
import getpass

from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


import json

from langchain_core.messages import (AIMessage,
                                    BaseMessage,
                                    ChatMessage,
                                    FunctionMessage,
                                    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
import functools


import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union



# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str



def init_llm_langsmith(llm_key = 3):
    """Initialize the LLM model for the LangSmith system."""
    # Set environment variables
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass(f"Please provide your {var}")
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_API_KEY")
    # Add tracing in LangSmith.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if llm_key == 3:
        llm = "gpt-3.5-turbo-0125"
        os.environ["LANGCHAIN_PROJECT"] = "GPT-3.5 Teaching Agent"
    elif llm_key == 4:
        llm = "gpt-4-0125-preview"
        os.environ["LANGCHAIN_PROJECT"] = "GPT-4 Teaching Agent"
    return llm






class TeachingAgent:
    """Class for the teaching agent system."""

    def __init__(self, llm: str="gpt-3.5-turbo-0125"):
        """Initialize the Teaching Agent system."""
        self.model = ChatOpenAI(model_name=llm, temperature=0.5)
        return None

    def agent_node(self, state, agent, name):
        """Helper function to create a node for a given agent. Node that invokes agent."""
        result = agent.invoke(state)
        # We convert the agent output into a format that is suitable to append to the global state
        if isinstance(result, FunctionMessage):
            pass
        else:
            result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            # Since we have a strict workflow, we can
            # track the sender so we know who to pass to next.
            "sender": name,
        }

    def create_frontline_node(self):
        """Create the frontline node."""
        system_message = """You are frontline teaching assistant.
                        Be concise in your responses.
                        You can chat with students and help them with basic questions, but if the student has a teaching-related question,
                        do not try to answer the question directly or gather information.
                        Instead, immediately transfer them to the teaching agent system by asking the user to hold for a moment.
                        Otherwise, just respond conversationally."""
        prompt = ChatPromptTemplate.from_messages([("system", system_message),
                                                   MessagesPlaceholder(variable_name="messages")])
        chain = prompt | self.model
        node = functools.partial(self.agent_node, agent=chain, name="Frontline")
        return node

    def create_teaching_agent_node(self):
        """Create the teaching agent node."""
        system_message = """You are a teaching assistant.
                        You are responsible for answering questions related to the course material.
                        You can use the tools provided to help you answer questions.
                        If you are unable to answer the question, transfer the student back to the frontline assistant."""
        prompt = ChatPromptTemplate.from_messages([("system", system_message),
                                                   MessagesPlaceholder(variable_name="messages")])
        chain = prompt | self.model
        node = functools.partial(self.agent_node, agent=chain, name="Teaching Agent")
        return node

    def create_graph(self):
        """Teaching Agent System Graph."""
        frontline_node = self.create_frontline_node()
        


        graph = StateGraph(AgentState)
        graph.add_node(frontline_node)

        graph.set_entry_point(frontline_node)
        return graph





if __name__ == "__main__":

    llm_ver = init_llm_langsmith(llm_key=3)

    teaching_agent_class = TeachingAgent(llm=llm_ver)
    teaching_agent_graph = teaching_agent_class.create_graph()
    print(teaching_agent_graph)

