# Basic imports
import os
import getpass

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




class TAS:
    def init(self,
             llm_model,
             version: str):
        """Initialize the Teaching Agent System."""
        self.llm_model = llm_model
        self.agent_prompt = self.build_agent_prompt()


    def build_agent_prompt(self):
        """Build the agent prompt."""
        system_message = """You will interact with a student who has no prior knowledge of the subject."""
        course = """Introduction to Computer Science"""
        subject = """Gradient Descent"""

        prompt_hub_template = hub.pull("augustsemrau/react-teaching-chat").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial(system_message=system_message, course_name=course, subject_name=subject)
        return prompt


    def build_baseline(self):
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
        tas_agent = create_react_agent(llm=llm_model, tools=tools, prompt=self.agent_prompt, output_parser=None)
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
        tas_agent = create_react_agent(llm=llm_model, tools=tools, prompt=self.agent_prompt, output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent, tools=tools, memory=tas_v1_memory, verbose=True, handle_parsing_errors=True)
        return tas_agent_executor



    def build_tas_v2(self):
        """Build the Teaching Agent System version 2.

        This version of the TAS is agenic, and uses complex tools such as other agents.
        """
        tools = []


        tas_v2_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        tas_agent = create_react_agent(llm=llm_model, tools=tools, prompt=self.agent_prompt, output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent, tools=tools, memory=tas_v2_memory, verbose=True, handle_parsing_errors=True)
        return tas_agent_executor



    def build_tas_v3(self):
        """Build the Teaching Agent System version 3.

        This version of the TAS is agenic, and uses very complex tools such as multi-agent systems.
        """
        tools = []


        tas_v3_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        tas_agent = create_react_agent(llm=llm_model, tools=tools, prompt=self.agent_prompt, output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent, tools=tools, memory=tas_v3_memory, verbose=True, handle_parsing_errors=True)
        return tas_agent_executor
























if __name__ == '__main__':
    def init_llm_langsmith(llm_key = 3, temp = 0.5):
        """Initialize the LLM model and LangSmith tracing."""
        # Set environment variables
        def _set_if_undefined(var: str):
            if not os.environ.get(var):
                os.environ[var] = getpass(f"Please provide your {var}")
        _set_if_undefined("OPENAI_API_KEY")
        _set_if_undefined("LANGCHAIN_API_KEY")

        # Add tracing in LangSmith.
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if llm_key == 3:
            llm_ver = "gpt-3.5-turbo-0125"
        elif llm_key == 4:
            llm_ver = "gpt-4-0125-preview"
        os.environ["LANGCHAIN_PROJECT"] = str(llm_ver + "Temp: " + temp + " TAS TEST 1")

        llm_model = ChatOpenAI(model_name=llm_ver, temperature=temp)
        return llm_model

    llm_model = init_llm_langsmith(llm_key=3, temp=0)
    # tools = []