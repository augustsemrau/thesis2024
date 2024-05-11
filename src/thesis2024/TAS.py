"""Teaching Agent System (TAS) for the thesis2024 project."""

import time

# Langchain imports
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
import chainlit as cl

# Tool imports
from langchain.tools import StructuredTool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper

# Local imports
from thesis2024.utils import init_llm_langsmith
from thesis2024.memory import LongTermMemory
from thesis2024.models.coding_agent import CodingMultiAgent
from thesis2024.datamodules.load_vectorstore import load_peristent_chroma_store



"""Tools for the Teaching Agent System (TAS) v1."""
class ToolClass:
    """Class for the tools in the Teaching Agent System."""

    def __init__(self):
        """Initialize the tool class."""
        pass

    """Internet Search Tool using DuckDuckGo API."""
    def build_search_tool(self):
        """Build the search tool."""
        # search_func = DuckDuckGoSearchAPIWrapper()
        # search_tool = Tool(name="Current Search",
        #                 func=search_func.run,
        #                 description="Useful when you need to answer questions about current events or the current state of the world."
        #                 )

        search_func = TavilySearchResults()
        search_tool = Tool(name="Web Search",
                        func=search_func.invoke,
                        description="Useful when you need to answer questions about current events or the current state of the world."
                        )
        return search_tool

    """Retrieval Tool using Chroma as vectorstore."""
    def build_retrieval_tool(self, course_name="Math1"):
        """Build the retrieval tool."""
        course_list = ["Mat1", "Math1", "DeepLearning", "IntroToMachineLearning"]
        if course_name not in course_list:
            raise ValueError(f"Course name not recognized. Should be one of {course_list}.")
        chroma_instance = load_peristent_chroma_store(openai_embedding=True, vectorstore_path=f"data/vectorstores/{course_name}")


        def retrieval_function(query: str):
            docs = chroma_instance.similarity_search(query, k = 3)
            if len(docs) == 0:
                return "No relevant documents found in all local data."
            else:
                # append the first 3 documents to the 
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

    """Coding Tool using Python REPL."""
    # TODO Coding tool should work, but is not implemented as it is not currently sandboxed correctly.
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

"""Agents for the Teaching Agent System (TAS) v2."""
class AgentClass:
    """Class for the agents used in the Teaching Agent System."""

    def __init__(self,
                 llm_model,
                 tool_class):
        """Initialize the agent class."""
        self.llm_model = llm_model
        self.tool_class = tool_class

    """Prompt for the Agent Tools."""
    def build_tool_agent_prompt(self):
        """Build the agent prompt."""
        prompt_hub_template = hub.pull("augustsemrau/react-agent-tool").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial()
        return prompt


    def build_search_agent(self):
        """Build the search agent."""
        tool_class = ToolClass()
        tools = [tool_class.build_search_tool()]

        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.build_tool_agent_prompt(),
                                       output_parser=None)
        # tas_v1_memory = self.init_memory()
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                        #    memory=tas_v1_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        def search_agent_function(query: str):
            """Search tool function."""
            output = tas_agent_executor.invoke({"input": query})["output"]
            return output
        search_agent = StructuredTool.from_function(
                                        name="Search Agent",
                                        func=search_agent_function,
                                        description="Useful when you need to answer questions about current events or the current state of the world."
                                        )
        return search_agent



    def build_retrieval_agent(self):
        """Build the retrieval agent."""
        retrieval_tool = self.tool_class.build_retrieval_tool()
        retrieval_agent = StructuredTool.from_function(
                            name="Retrieval Agent",
                            func=retrieval_tool.invoke,
                            description="Useful when you need to answer questions using relevant course material."
                            )
        return retrieval_agent

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
                 course: str = "Math1",
                 student_name=None,
                 use_longterm_memory=False,
                 ):
        """Initialize the Teaching Agent System."""
        self.llm_model = llm_model
        self.course = course
        self.use_longterm_memory = use_longterm_memory
        # Init short term memory for the TAS
        self.short_term_memory = ConversationBufferMemory(memory_key="chat_history",
                                                          return_messages=False,
                                                          ai_prefix="Teaching Assistant",
                                                          human_prefix="Student")
        # Init long term memory for the TAS
        self.student_name = student_name # If student_name is None, the TAS will not use long-term memory
        if student_name is not None:
            self.long_term_memory_class = LongTermMemory(user_name=self.student_name)
        self.tas_prompt = self.build_tas_prompt()
        self.build_executor(ver=version)

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
        if self.use_longterm_memory:
            facts = self.long_term_memory_class.get_user_memories(query="")
            # TODO Advanced memory types such as core_beliefs, formative_events, longterm_memory
            prompt_hub_template = hub.pull("augustsemrau/react-langmem-teaching-chat").template
            prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
            prompt = prompt_template.partial(course_name=self.course,
                                             facts = facts,
                                            # core_beliefs=core_beliefs,
                                            # formative_events=formative_events,
                                            # longterm_memory=longterm_memory,
                                            )
        else:
            prompt_hub_template = hub.pull("augustsemrau/react-teaching-chat").template
            prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
            prompt = prompt_template.partial(course_name=self.course)
        return prompt


    """TAS v0 has one agent with no tools."""
    def build_tas_v0(self):
        """Build the Teaching Agent System version 0.

        This version of the TAS is agenic, but has no tools.
        """
        tools = []  # NO TOOLS FOR v0

        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.tas_prompt,
                                       output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                           memory=self.short_term_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        return tas_agent_executor
    """v0 is DONE"""

    """TODO TAS v1 has one agent using tools."""
    def build_tas_v1(self):
        """Build the Teaching Agent System version 1.

        This version of the TAS is agenic, and has simple tools.
        """
        tool_class = ToolClass()
        tools = [tool_class.build_search_tool(),
                 tool_class.build_retrieval_tool(course_name=self.course),
                 tool_class.build_coding_tool(),
                 tool_class.build_math_tool()]

        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.tas_prompt,
                                       output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                           memory=self.short_term_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        return tas_agent_executor

    """TODO TAS v2 has one agent using other agents which have access to tools."""
    def build_tas_v2(self):
        """Build the Teaching Agent System version 2.

        This version of the TAS is agenic, and uses complex tools such as other agents.
        """
        agent_class = AgentClass(llm_model=self.llm_model, tool_class=ToolClass())
        search_agent = agent_class.build_search_agent()
        tools = [search_agent]

        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.tas_prompt,
                                       output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                           memory=self.short_term_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        return tas_agent_executor

    """TODO TAS v3 has one agent using multi-agent systems."""
    def build_tas_v3(self):
        """Build the Teaching Agent System version 3.

        This version of the TAS is agenic, and uses very complex tools such as multi-agent systems.
        """
        multi_agent_class = MultiAgentClass(llm_model=self.llm_model)
        coding_multi_agent = multi_agent_class.build_coding_multi_agent()
        tools = [coding_multi_agent]

        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.tas_prompt,
                                       output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                           memory=self.short_term_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        return tas_agent_executor



    """Predict function for invoking the initiated TAS."""
    def predict(self, query):
        """Invoke the Teaching Agent System."""
        print("\n\nUser Query:", query)
        response = self.tas_executor.invoke({"input": query})[self.output_tag]
        print("\n\nTAS Memory:")
        print(f"\n{self.tas_executor.memory}\n")
        if self.student_name is not None:
            self.long_term_memory_class.save_conversation_step(user_query=query, llm_response=response)
        return response



    """(Likely not for use) Baseline Teaching Agent System (will maybe be redundant)."""
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

    """(DOES NOT WORK) Predict function for invoking the initiated TAS asynchronously for use in Chainlit frontend."""
    def cl_predict(self, query):
        """Invoke the Teaching Agent System."""

        res = self.tas_executor.invoke({"input": query}, callbacks=[cl.AsyncLangchainCallbackHandler()])
        return res[self.output_tag]










if __name__ == '__main__':

    # test_version = "baseline"
    test_version = "v1"
    time_now = time.strftime("%Y.%m.%d-%H.%M.")
    langsmith_name = test_version + " TAS TEST " + time_now
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name=langsmith_name)

    tas = TAS(llm_model=llm_model,
              version=test_version,
              course="IntroToMachineLearning",
              student_name="AugustSemrau1",
              use_longterm_memory=False,
              )

    res = tas.predict("Hello, I am August! Today, I would like to learn about the three most important supervised learning algorithms.")
    print("\n\nResponse: ", res)
    res = tas.predict("What is the name of the person who invented the ADAM optimization technique?")
    print("\n\nResponse: ", res)
    res = tas.predict("Thank you for the help, have a nice day!")
    print("\n\nResponse: ", res)




    # Old test queries
    # res = tas.predict("What is the current world record in 50 meter butterfly?")
    # res = tas.predict("Can you explain me how ADAM optimization works?")
    # print(tas.predict("Hej! Jeg vil gerne snakke dansk. Kan du forklare mig hvordan lineær regression virker?"))#["output"]
