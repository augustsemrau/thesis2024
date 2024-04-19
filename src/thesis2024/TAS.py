"""Teaching Agent System (TAS) for the thesis2024 project."""

# Langchain imports
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, messages_to_dict
from langchain.chains import ConversationChain, LLMChain
import chainlit as cl

# Tool imports
from langchain.tools import StructuredTool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
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

    def build_retrieval_tool(self, course_name="Matematik1"):
        """Build the retrieval tool."""
        course_list = ["Matematik 1", "Mathematics 1", "Deep Learning"]
        if course_name not in course_list:
            raise ValueError(f"Course name not recognized. Should be one of {course_list}.")

        if course_name == "Matematik1":
            chroma_instance = load_peristent_chroma_store(openai_embedding=True, vectorstore_path="data/vectorstores/Matematik1")
        elif course_name == "Mathematics 1":
            chroma_instance = load_peristent_chroma_store(openai_embedding=True, vectorstore_path="data/vectorstores/Math1_new")
        elif course_name == "Deep Learning":
            chroma_instance = load_peristent_chroma_store(openai_embedding=True, vectorstore_path="data/vectorstores/DeepLearning")

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

    def __init__(self,
                 llm_model,
                 tool_class):
        """Initialize the agent class."""
        self.llm_model = llm_model
        self.tool_class = tool_class

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
                 version: str = "v0",
                 course: str = "Mathematics 1",
                 subject: str = "All subjects"):
        """Initialize the Teaching Agent System."""
        self.llm_model = llm_model
        self.course = course
        self.subject = subject
        self.tas_prompt = self.build_tas_prompt()
        self.build_executor(ver=version)

    """Initialize the memory for the Teaching Agent System."""
    def init_memory(self):
        """Initialize the memory for the Teaching Agent System."""
        memory = ConversationBufferMemory(memory_key="chat_history",
                                              return_messages=False,
                                              ai_prefix="Teaching Assistant",
                                              human_prefix="Student",
                                              )
        return memory

    """Build the Teaching Agent System executor."""
    def build_executor(self, ver):
        """Build the Teaching Agent System executor."""
        self.output_tag = "output"
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
            self.output_tag = "response"

    """Prompt for the Teaching Agent System."""
    def build_tas_prompt(self):
        """Build the agent prompt."""
        system_message = """You will interact with a student who has no prior knowledge of the subject."""
        prompt_hub_template = hub.pull("augustsemrau/react-teaching-chat").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial(system_message=system_message,
                                         course_name=self.course,
                                         subject_name=self.subject)
        return prompt

    """Baseline Teaching Agent System (will maybe be redundant)."""
    def build_nonagenic_baseline(self):
        """Build the baseline Teaching Agent System."""
        prompt = {
            "chat_history": {},
            "input": input,
            "system_message": ".",
        }
        prompt_template = """You are a teaching assistant. You are responsible for answering questions and inqueries that the student might have.\n
Here is the student's query, which you MUST respond to:
{input}\n
This is the conversation so far:
{chat_history}"""
        prompt = PromptTemplate.from_template(template=prompt_template)
        #  prompt = prompt_template.partial(system_message=system_message, course_name=course, subject_name=subject)
        baseline_memory = self.init_memory()
        baseline_chain = ConversationChain(llm=self.llm_model,
                                prompt=prompt,
                                memory=baseline_memory,
                                #output_parser=BaseLLMOutputParser(),
                                verbose=False,)
        return baseline_chain

    """TAS v0 has one agent with no tools."""
    def build_tas_v0(self):
        """Build the Teaching Agent System version 0.

        This version of the TAS is agenic, but has no tools.
        """
        tools = []  # NO TOOLS FOR v0
        tas_v0_memory = self.init_memory()
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

    """TAS v1 has one agent using tools."""
    def build_tas_v1(self):
        """Build the Teaching Agent System version 1.

        This version of the TAS is agenic, and has simple tools.
        """
        tool_class = ToolClass()
        tools = [tool_class.build_search_tool(), tool_class.build_retrieval_tool(course_name=self.course)]#, tool_class.build_coding_tool()]

        tas_v1_memory = self.init_memory()
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

    """TAS v2 has one agent using other agents which have access to tools."""
    def build_tas_v2(self):
        """Build the Teaching Agent System version 2.

        This version of the TAS is agenic, and uses complex tools such as other agents.
        """
        agent_class = AgentClass(llm_model=self.llm_model)
        search_agent = agent_class.build_search_agent()
        tools = [search_agent]


        tools = []
        tas_v2_memory = self.init_memory()
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

    """TAS v3 has one agent using multi-agent systems."""
    def build_tas_v3(self):
        """Build the Teaching Agent System version 3.

        This version of the TAS is agenic, and uses very complex tools such as multi-agent systems.
        """
        multi_agent_class = MultiAgentClass(llm_model=self.llm_model)
        coding_multi_agent = multi_agent_class.build_coding_multi_agent()
        tools = [coding_multi_agent]

        tas_v3_memory = self.init_memory()
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

    """Predict function for invoking the initiated TAS."""
    def predict(self, query):
        """Invoke the Teaching Agent System."""
        print("\n\nUser Query:", query)
        response = self.tas_executor.invoke({"input": query})[self.output_tag]
        print("\n\nTAS Memory:")
        print(self.tas_executor.memory)#.chat_memory.messages)
        # print(messages_to_dict(self.tas_executor.memory.chat_memory.messages))
        return response

    """(DOES NOT WORK) Predict function for invoking the initiated TAS asynchronously for use in Chainlit frontend."""
    def cl_predict(self, query):
        """Invoke the Teaching Agent System."""
        if self.agenic:
            res = self.tas_executor.invoke({"input": query}, callbacks=[cl.AsyncLangchainCallbackHandler()])
            return res["output"]
        else:
            res = self.tas_executor.invoke({"input": query}, callbacks=[cl.AsyncLangchainCallbackHandler()])
            return res["text"]









if __name__ == '__main__':

    # test_version = "baseline"
    test_version = "Baseline"
    langsmith_name = "TAS TEST 1 " + test_version
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name=langsmith_name)

    tas = TAS(llm_model=llm_model, version=test_version)

    # print(tas.predict("Hej! Jeg vil gerne snakke dansk. Kan du forklare mig hvordan lineær regression virker?"))#["output"]

    res = tas.predict("Hello, I am August!")
    print("\n\nResponse: ", res)
    res = tas.predict("Can you explain me how ADAM optimization works?")
    print("\n\nResponse: ", res)
    tas.predict("What is the name of the person who invented this optimization technique?")
