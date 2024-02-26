"""Corrective _RAG (CRAG) model for optimized retrieval."""
# Imports for api keys
import getpass
import os
import uuid

# Imports for retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Imports for state
from typing import Dict, TypedDict
from langchain_core.messages import BaseMessage

# Imports for CRAG class
import json
import operator
from typing import Annotated, Sequence, TypedDict

from langchain import hub
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Imports for building graph
import pprint
from langgraph.graph import END, StateGraph

# Imports for running this file
from thesis2024.datamodules.load_vectorstore import load_peristent_chroma_store





class GraphState(TypedDict):
    """Represents the state of our graph.

    Attributes
    ----------
        keys: A dictionary where each key is a string.

    """

    keys: Dict[str, any]



class CragNodes():
    """Represents the nodes of our graph.

    Attributes
    ----------
        retrieve: A function that retrieves documents.
        generate: A function that generates an answer.
        grade_documents: A function that grades documents.
        transform_query: A function that transforms the query.
        web_search: A function that performs a web search.

    """

    def __init__(self,
                 generate_model: str="gpt-3.5-turbo",
                 grade_model: str="gpt-4-0125-preview",
                 transform_query_model: str="gpt-4-0125-preview"
                 ):
        """Initialize the CRAG nodes.

        Args:
        ----
            gpt_model (str): The GPT model to use for generation.

        """
        self.generate_model = generate_model
        self.grade_model = grade_model
        self.transform_query_model = transform_query_model
        return None


    def retrieve(self, state):
        """CRAG Node Function: Retrieve documents.

        Args:
        ----
            state (dict): The current graph state

        Returns:
        -------
            state (dict): New key added to state, documents, that contains retrieved documents

        """
        print("---RETRIEVE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = retriever.get_relevant_documents(question)
        return {"keys": {"documents": documents, "question": question}}


    def generate(self, state):
        """CRAG Node Function: Generate answer.

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


    def grade_documents(self, state):
        """CRAG Node Function: Determine whether the retrieved documents are relevant to the question.

        Args:
        ----
            state (dict): The current graph state

        Returns:
        -------
            state (dict): Updates documents key with relevant documents

        """
        print("---CHECK RELEVANCE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM
        model = ChatOpenAI(temperature=0, model=self.grade_model, streaming=True)

        # Tool
        grade_tool_oai = convert_to_openai_tool(grade)

        # LLM with tool and enforce invocation
        llm_with_tool = model.bind(
            tools=[convert_to_openai_tool(grade_tool_oai)],
            tool_choice={"type": "function", "function": {"name": "grade"}},
        )

        # Parser
        parser_tool = PydanticToolsParser(tools=[grade])

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        # Chain
        chain = prompt | llm_with_tool | parser_tool

        # Score
        filtered_docs = []
        search = "No"  # Default do not opt for web search to supplement retrieval
        for d in documents:
            score = chain.invoke({"question": question, "context": d.page_content})
            grade = score[0].binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                search = "Yes"  # Perform web search
                continue

        return {
            "keys": {
                "documents": filtered_docs,
                "question": question,
                "run_web_search": search,
            }
        }


    def transform_query(self, state):
        """CRAG Node Function: Transform the query to produce a better question.

        Args:
        ----
            state (dict): The current graph state

        Returns:
        -------
            state (dict): Updates question key with a re-phrased question

        """
        print("---TRANSFORM QUERY---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        # Create a prompt template with format instructions and the query
        prompt = PromptTemplate(
            template="""You are generating questions that is well optimized for retrieval. \n 
            Look at the input and try to reason about the underlying sematic intent / meaning. \n 
            Here is the initial question:
            \n ------- \n
            {question} 
            \n ------- \n
            Formulate an improved question: """,
            input_variables=["question"],
        )

        # Grader
        model = ChatOpenAI(temperature=0, model=self.transform_query_model, streaming=True)

        # Prompt
        chain = prompt | model | StrOutputParser()
        better_question = chain.invoke({"question": question})

        return {"keys": {"documents": documents, "question": better_question}}


    def web_search(self, state):
        """CRAG Node Function: Web search based on the re-phrased question using Tavily API.

        Args:
        ----
            state (dict): The current graph state

        Returns:
        -------
            state (dict): Updates documents key with appended web results

        """
        print("---WEB SEARCH---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        tool = TavilySearchResults()
        docs = tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"keys": {"documents": documents, "question": question}}



    def decide_to_generate(self, state):
        """CRAG Edge Function: Determine whether to generate an answer or re-generate a question for web search.

        Args:
        ----
            state (dict): The current state of the agent, including all keys.

        Returns:
        -------
            str: Next node to call

        """
        print("---DECIDE TO GENERATE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        filtered_documents = state_dict["documents"]
        search = state_dict["run_web_search"]

        if search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"



### Function for building graph

def build_graph(node_class):
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





    # Load vectorstore
    vectorstore = load_peristent_chroma_store(openai_embedding=True)
    retriever = load_peristent_chroma_store(openai_embedding=True).as_retriever()

    crag_nodes = CragNodes()
    # Build graph
    app = build_graph(node_class=crag_nodes)


    # Run
    inputs = {"keys": {"question": "What are some aspects of machine learning that are important for a data scientist to know?"}}
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    pprint.pprint(value["keys"]["generation"])


