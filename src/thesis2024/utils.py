"""Module containing the utility functions for the thesis project."""

# GraphState
from typing import Dict, TypedDict

# AgentState
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """Represents the state of our graph.

    Attributes
    ----------
        keys: A dictionary where each key is a string.

    """

    keys: Dict[str, any]


class AgentState(TypedDict):
    """Represents the state of our agent.

    Attributes
    ----------
        messages: A sequence of messages.

    """

    messages: Annotated[Sequence[BaseMessage], operator.add]

