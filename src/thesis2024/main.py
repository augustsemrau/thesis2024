"""Main script for the thesis project."""

import os
import sys

## Langchain imports
from langgraph.graph import END, StateGraph

## Local imports
from thesis2024.utils import GraphState
from thesis2024.datamodules.crag import CragNodes
from thesis2024.datamodules.retrieval_agent import RetrievalAgent








### Function for building graph
def build_rag_graph(node_class):
    """Build the graph for the CRAG model."""
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", node_class.retrieve)  # retrieve
    workflow.add_node("grade_documents", node_class.grade_documents)  # grade documents
    workflow.add_node("generate", node_class.generate)  # generatae
    workflow.add_node("transform_query", node_class.transform_query)  # transform_query
    workflow.add_node("web_search", node_class.web_search)  # web search

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        node_class.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()
    return app




if __name__ == "__main__":
    # Build the graph
    app = build_rag_graph()
    print(app)
    # Run the graph
    app.run()
    print("Graph run complete.")
    sys.exit(0)