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
from langchain.utilities import DuckDuckGoSearchAPIWrapper

# Local imports
from thesis2024.models.coding_agent import CodingMultiAgent
from thesis2024.utils import init_llm_langsmith



class TAS:
    """Class for the Teaching Agent System."""

    def __init__(self,
                 llm_model,
                 version: str = "v0"):
        """Initialize the Teaching Agent System."""
        self.llm_model = llm_model
        self.tas_prompt = self.build_tas_prompt()

        if version == "v0":
            self.tas_executor = self.build_tas_v0()
        elif version == "v1":
            self.tas_executor = self.build_tas_v1()
        elif version == "v2":
            self.tas_executor = self.build_tas_v2()
        elif version == "v3":
            self.tas_executor = self.build_tas_v3()


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
        tas_agent = create_react_agent(llm=self.llm_model, tools=tools, prompt=self.tas_prompt, output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent, tools=tools, memory=tas_v0_memory, verbose=True, handle_parsing_errors=True)
        return tas_agent_executor



    def build_tas_v1(self):
        """Build the Teaching Agent System version 1.

        This version of the TAS is agenic, and has simple tools.
        """
        """Search tool."""
        search = DuckDuckGoSearchAPIWrapper()
        search_tool = Tool(name="Current Search",
                        func=search.run,
                        description="Useful when you need to answer questions about nouns, current events or the current state of the world."
                        )




        tools = [search_tool]
        tas_v1_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        tas_agent = create_react_agent(llm=self.llm_model, tools=tools, prompt=self.tas_prompt, output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent, tools=tools, memory=tas_v1_memory, verbose=True, handle_parsing_errors=True)
        return tas_agent_executor



    def build_tas_v2(self):
        """Build the Teaching Agent System version 2.

        This version of the TAS is agenic, and uses complex tools such as other agents.
        """



        tools = []
        tas_v2_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        tas_agent = create_react_agent(llm=self.llm_model, tools=tools, prompt=self.tas_prompt, output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent, tools=tools, memory=tas_v2_memory, verbose=True, handle_parsing_errors=True)
        return tas_agent_executor



    def build_tas_v3(self):
        """Build the Teaching Agent System version 3.

        This version of the TAS is agenic, and uses very complex tools such as multi-agent systems.
        """
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


        tools = [coding_multiagent]
        tas_v3_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        tas_agent = create_react_agent(llm=self.llm_model, tools=tools, prompt=self.tas_prompt, output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent, tools=tools, memory=tas_v3_memory, verbose=True, handle_parsing_errors=True)
        return tas_agent_executor

    def predict(self, query):
        """Invoke the Teaching Agent System."""
        response = self.tas_executor.invoke({"input": query})["output"]
        return response









if __name__ == '__main__':
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="TAS TEST 1")

    tas = TAS(llm_model=llm_model)
    tas_v0 = tas.build_tas_v0()

    print(tas_v0.invoke({"input": "Hello I am August?",}))#["output"]
