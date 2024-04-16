"""Teaching Agent System (TAS) for the thesis2024 project."""

# Langchain imports
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Tool imports
from langchain.tools import StructuredTool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL

# Local imports
from thesis2024.utils import init_llm_langsmith
from thesis2024.models.coding_agent import CodingMultiAgent
from thesis2024.datamodules.load_vectorstore import load_peristent_chroma_store


"""Tools for the Teaching Agent System (TAS) v1."""
class ToolClass:
    """Class for the tools in the Teaching Agent System."""

    def __init__(self):
        """Initialize the tool class."""
        pass

    def build_search_tool(self):
        """Build the search tool."""
        search = DuckDuckGoSearchAPIWrapper()
        search_tool = Tool(name="Current Search",
                        func=search.run,
                        description="Useful when you need to answer questions about nouns, current events or the current state of the world."
                        )
        return search_tool

    def build_retrieval_tool(self):
        """Build the retrieval tool."""
        chroma_instance = load_peristent_chroma_store(openai_embedding=True,
                                                        vectorstore_path="data/processed/chroma")
        def retrieval_function(query: str):
            docs = chroma_instance.similarity_search(query, k = 1)
            return docs[0].page_content
        retrieval_tool = StructuredTool.from_function(
                            name="Retrieval Tool",
                            func=retrieval_function,
                            description="Useful when you need to answer questions using relevant course material."
                            )
        return retrieval_tool

    ## Coding tool should work, but is not implemented as it is not currently sandboxed correctly.
    def build_coding_tool(self):
        """Build a coding tool."""
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
                            description="Useful when you need to answer questions using code."
                            )
        return coding_tool

"""Agents for the Teaching Agent System (TAS) v2."""
class AgentClass:
    """Class for the agents used in the Teaching Agent System."""

    def __init__(self):
        """Initialize the agent class."""
        pass

    def build_search_agent(self):
        """Build the search agent."""
        search_agent = 1
        return search_agent

"""Multi-Agent systems for the Teaching Agent System (TAS) v3."""
class MultiAgentClass:
    """Class for the multi-agent systems used in the Teaching Agent System."""

    def __init__(self, llm_model):
        """Initialize the agent class."""
        self.llm_model = llm_model


    def build_coding_multi_agent(self):
        """Coding multi-agent as a tool."""
        coding_subgraph_class = CodingMultiAgent(llm_model=self.llm_model)
        coding_graph = coding_subgraph_class.instanciate_graph()
        def coding_function(query: str):
            """Coding tool function."""
            output = coding_graph.invoke({"messages": [HumanMessage(content=query)]},
                                        {"recursion_limit": 100})
            return output["messages"][-1].content
        coding_multiagent = StructuredTool.from_function(
                                    func=coding_function,
                                    name="Coding Tool",
                                    description="Useful when you need to answer questions using a coded example."
                                    )
        return coding_multiagent

"""Teaching Agent System (TAS) for the thesis2024 project."""
class TAS:
    """Class for the Teaching Agent System."""

    def __init__(self,
                 llm_model,
                 version: str = "v0"):
        """Initialize the Teaching Agent System."""
        self.llm_model = llm_model
        self.tas_prompt = self.build_tas_prompt()
        self.build_executor(ver=version)

    def build_executor(self, ver):
        """Build the Teaching Agent System executor."""
        self.agenic = True
        if ver == "v0":
            self.tas_executor = self.build_tas_v0()
        elif ver == "v1":
            self.tas_executor = self.build_tas_v1()
        elif ver == "v2":
            self.tas_executor = self.build_tas_v2()
        elif ver == "v3":
            self.tas_executor = self.build_tas_v3()
        else:
            self.tas_executor = self.build_nonagenic_baseline()
            self.agenic = False


    def build_tas_prompt(self):
        """Build the agent prompt."""
        system_message = """You will interact with a student who has no prior knowledge of the subject."""
        course = """Introduction to Computer Science"""
        subject = """Gradient Descent"""

        prompt_hub_template = hub.pull("augustsemrau/react-teaching-chat").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial(system_message=system_message, course_name=course, subject_name=subject)
        return prompt


    def build_nonagenic_baseline(self):
        """Build the baseline Teaching Agent System."""
        prompt = "You are a teaching assistant. You are responsible for answering questions related to the course material."
        chain = self.llm_model | prompt
        return chain

    def build_tas_v0(self):
        """Build the Teaching Agent System version 0.

        This version of the TAS is agenic, but has no tools.
        """
        tools = []  # NO TOOLS FOR v0
        tas_v0_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.tas_prompt,
                                       output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                           memory=tas_v0_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        return tas_agent_executor

    def build_tas_v1(self):
        """Build the Teaching Agent System version 1.

        This version of the TAS is agenic, and has simple tools.
        """
        tool_class = ToolClass()
        tools = [tool_class.build_search_tool(), tool_class.build_retrieval_tool()]#, tool_class.build_coding_tool()]

        tas_v1_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.tas_prompt,
                                       output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                           memory=tas_v1_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        return tas_agent_executor

    def build_tas_v2(self):
        """Build the Teaching Agent System version 2.

        This version of the TAS is agenic, and uses complex tools such as other agents.
        """
        agent_class = AgentClass(llm_model=self.llm_model)
        search_agent = agent_class.build_search_agent()
        tools = [search_agent]


        tools = []
        tas_v2_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.tas_prompt,
                                       output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                           memory=tas_v2_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        return tas_agent_executor

    def build_tas_v3(self):
        """Build the Teaching Agent System version 3.

        This version of the TAS is agenic, and uses very complex tools such as multi-agent systems.
        """
        multi_agent_class = MultiAgentClass(llm_model=self.llm_model)
        coding_multi_agent = multi_agent_class.build_coding_multi_agent()
        tools = [coding_multi_agent]

        tas_v3_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.tas_prompt,
                                       output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                           memory=tas_v3_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        return tas_agent_executor

    def predict(self, query):
        """Invoke the Teaching Agent System."""
        if not self.agenic:
            response = self.tas_executor.invoke({"input": query})["text"]
        else:
            response = self.tas_executor.invoke({"input": query})["output"]
        return response









if __name__ == '__main__':
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="TAS v1 TEST 1")

    tas = TAS(llm_model=llm_model, version="v1")

    print(tas.predict("Hej! Jeg vil gerne snakke dansk. Kan du forklare mig hvordan lineær regression virker?"))#["output"]
