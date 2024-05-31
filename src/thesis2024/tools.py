"""Tools for the all systems in this project."""

# Python Imports
from typing import Annotated

# Langchain Imports
from langchain.agents import Tool
from langchain.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper

# Local Imports
from thesis2024.datamodules.load_vectorstore import load_peristent_chroma_store
from thesis2024.multiagent_modules.crag import Crag




class ToolClass:
    """Class for the tools in the Teaching Agent System."""

    def __init__(self):
        """Initialize the tool class."""
        pass


    def build_search_tool(self):
        """Internet Search Tool using Tavily API."""
        search_func = TavilySearchResults()
        search_tool = Tool(name="Web Search",
                        func=search_func.invoke,
                        description="Useful when you need to answer questions about current events or the current state of the world."
                        )
        return search_tool


    def build_retrieval_tool(self, course_name="Math1"):
        """RAG Tool using Chroma as vectorstore."""
        course_list = ["Mat1", "Math1", "DeepLearning", "IntroToMachineLearning"]
        if course_name not in course_list:
            raise ValueError(f"Course name not recognized. Should be one of {course_list}.")
        chroma_instance = load_peristent_chroma_store(openai_embedding=True, vectorstore_path=f"data/vectorstores/{course_name}")


        def retrieval_function(query: str):
            docs = chroma_instance.similarity_search(query, k = 3)
            if len(docs) == 0:
                return "No relevant documents found in all local data."
            else:
                # append the first 3 documents to the tool return
                return_docs = ""
                for doc in docs:
                    return_docs += doc.page_content + "\n\n"
                return return_docs

        retrieval_tool = StructuredTool.from_function(
                            name="Retrieval Tool",
                            func=retrieval_function,
                            description="Useful when you need to answer questions using relevant course material."
                            )
        return retrieval_tool

    def build_crag_tool(self, course_name="IntroToMachineLearning"):
        """CRAG Tool."""
        crag_class = Crag(course_name=course_name,
                generate_model="gpt-3.5-turbo",
                grade_model="gpt-4-0125-preview",
                transform_query_model="gpt-4-0125-preview",
                vectorstore_dir="data/processed/chroma")

        def crag_tool(search_query: str):
            """Call the CRAG model to answer questions."""
            _, answer = crag_class.predict(question=search_query)
            return str(answer)

        crag_tool = StructuredTool.from_function(
                            name="CRAG Tool",
                            func=crag_tool,
                            description="Useful when you need to answer questions using relevant course material.",
                            return_direct=True
                            )
        return crag_tool



    def build_coding_tool(self):
        """Coding Tool using Python REPL."""
        repl = PythonREPL()
        def python_repl(
            code: Annotated[str, "The python code to execute to generate whatever fits the user needs."]
        ):
            """Use this to execute python code.

            If you want to see the output of a value,
            you should print it out with `print(...)`. This is visible to the user.
            """
            try:
                result = repl.run(code)
            except BaseException as e:
                return f"Failed to execute. Error: {repr(e)}"
            return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"

        coding_tool = StructuredTool.from_function(
                            name="Coding Tool",
                            func=python_repl,
                            description="Useful when you have some code you want to execute, for generating a plot for example."
                            )
        return coding_tool

    """Math tool for writing correct math formulas."""
    # TODO Make sure this tool works as intended.
    def build_math_tool(self):
        """Build a math tool."""
        math_func = WolframAlphaAPIWrapper()
        math_tool = Tool(name="Math Tool",
                        func=math_func.run,
                        description="Useful when you need to write and compute correct math formulas."
                        )
        return math_tool

