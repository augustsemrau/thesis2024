"""Mmodule containings the hierarchical multi-agent framework."""

# Basic Imports
import time
import datetime

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

# LangGraph Imports
from typing import List
from langgraph.graph import END, MessageGraph


# Local Imports
from thesis2024.utils import init_llm_langsmith

# Tools Imports
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode



class ToolClass:
    def __init__(self):
        self.scape_webpages = self.setup_scape_webpages()

    def setup_scape_webpages(self):
        @tool
        def scrape_webpages(urls: List[str]) -> str:
            """Use requests and bs4 to scrape the provided web pages for detailed information."""
            loader = WebBaseLoader(urls)
            docs = loader.load()
            return "\n\n".join(
                [
                    f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
                    for doc in docs
                ]
            )
        return scrape_webpages


class HierarchicalMultiAgent:
    """Class for Hierarchical Multi-Agent System."""

    def __init__(self, llm_model, tool_class):
        """Initialize the Hierarchical Multi-Agent System."""
        self.llm_model = llm_model
        self.tool_class = tool_class
        self.graph = self.build_graph()


    def build_graph(self):
        """Build the hierarchical message graph."""
        graph = MessageGraph()

        return graph

    def predict(self, query: str):
        """Predict the next message in the conversation."""
        response = []
        for s in self.graph.stream(
                {
                    "messages": [
                        HumanMessage(content=query)
                    ],
                },
                {"recursion_limit": 150},
            ):
            if "__end__" not in s:
                print(s)
                print("---")
                response.append(s)

        return response[-1]


if __name__ == "__main__":

    llm_model = init_llm_langsmith(llm_key=4, temp=0.5, langsmith_name="Hierarchical Test")





