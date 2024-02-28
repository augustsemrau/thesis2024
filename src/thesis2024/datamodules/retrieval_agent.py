"""Agent that determines whether to retrieve data, and if so, uses crag to retrieve data."""

# Basic imports
import getpass
import os
import pprint

# Langchain imports
from langgraph.graph import END, StateGraph

# Local imports
from thesis2024.utils import GraphState, AgentState
from thesis2024.datamodules.crag import Crag


import json
import operator
from typing import Annotated, Sequence, TypedDict

from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolInvocation
from langchain_core.output_parsers import StrOutputParser








if __name__ == "__main__":

    # Set environment variables
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass(f"Please provide your {var}")
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")
    # Add tracing in LangSmith.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Langgraph CRAG GPT-4"



    CragClass = Crag()
    # Build graph
    app = CragClass.build_rag_graph()


    # Run
    inputs = {"keys": {"question": "Who is the teacher of the machine learning course, and how come the highest mountains are located in asia?"}}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    pprint.pprint(value["keys"]["generation"])



