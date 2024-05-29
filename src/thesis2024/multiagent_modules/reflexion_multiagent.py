"""Mmodule containings the reflexion agent framework."""

# Langchain imports
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.react.output_parser import ReActOutputParser
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
from thesis2024.PMAS import LongTermMemory
from thesis2024.multiagent_modules.coding_agent import CodingMultiAgent
from thesis2024.datamodules.load_vectorstore import load_peristent_chroma_store



"""Tools for the Teaching Agent System (TAS) v1."""
class ToolClass:
    """Class for the tools in the Teaching Agent System."""

    def __init__(self):
        """Initialize the tool class."""
        pass

    """Internet Search Tool using Tavily API."""
    def build_search_tool(self):
        """Build the search tool."""
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


class ReflexionTool:
    """Class for the subgraph of the TAS."""
    def __init__(self,
                 llm_model,
                 max_iterations: int = 3,
                 course_name: str = "IntroToMachineLearning"):
        """Initialize the subgraph."""
        self.llm_model = llm_model
        self.max_iter = max_iterations
        self.course = course_name

        tool_class = ToolClass()
        self.tools = [tool_class.build_search_tool(),
                    tool_class.build_retrieval_tool(course_name=self.course),
                    tool_class.build_coding_tool(),
                    ]
        self.general_prompt = "You are working together with a several other agents in a teaching assistance system.\n\n"



    def initial_draft(self, query: str):
        """Initial draft of the subgraph."""
        specific_prompt = f"""You are responsible for creating the first draft of the response to the student's query:\n\n{query}\n\nBegin!"""
        prompt = self.general_prompt + specific_prompt

        return self.llm_model.invoke(prompt)


    def critique_draft(self, draft: str):
        """Critique the draft."""
        pass
    def new_draft(self, critique: str):
        """Create a new draft."""
        pass

    def finalize_draft(self, final_draft: str, final_critique: str):
        """Finalize the draft."""
        pass

    def build_nonagenic_baseline(self):
        """Baseline LLM Chain Teaching System."""
        prompt = {
            "chat_history": {},
            "input": input,
            "system_message": ".",
        }
        prompt_template = """You are a teaching assistant. You are responsible for answering questions and inquiries that the student might have.
Here is the student's query, which you MUST respond to:
{input}
This is the conversation so far:
{chat_history}"""
        prompt = PromptTemplate.from_template(template=prompt_template)
        #  prompt = prompt_template.partial(system_message=system_message, course_name=course, subject_name=subject)
        baseline_chain = ConversationChain(llm=self.llm_model,
                                prompt=prompt,
                                memory=self.short_term_memory,
                                #output_parser=BaseLLMOutputParser(),
                                verbose=False,)
        return baseline_chain

    def build_reflexion_tool(self):
        """Build the subgraph tool."""

        def invoke_reflexion_graph(self, query: str):
            """Invoke the Reflexion Subgraph."""
            initial_draft = self.initial_draft(query)
            critique = self.critique_draft(draft=initial_draft)
            for i in range(self.max_iter):
                new_draft = self.new_draft(critique=critique)
                critique = self.critique_draft(draft=new_draft)
            response = self.finalize_draft(final_draft=new_draft, final_critique=critique)
            return response

        reflexion_tool = StructuredTool.from_function(
                            name="Reflexion Tool",
                            func=invoke_reflexion_graph,
                            description="Useful when you need to answer questions."
                            )
        return reflexion_tool




"""Teaching Agent System (TAS) for the thesis2024 project."""
class ReflexionMultiAgent:
    """Class for the Teaching Agent System."""

    def __init__(self,
                 llm_model,
                 course: str = "IntroToMachineLearning",
                 ):
        """Initialize the Teaching Agent System."""
        self.llm_model = llm_model
        self.course = course

        # Init short term memory for the TAS
        self.short_term_memory = ConversationBufferMemory(memory_key="chat_history",
                                                          return_messages=False,
                                                          ai_prefix="Teaching Assistant",
                                                          human_prefix="Student")

        self.reflexion_prompt = self.build_reflexion_prompt()
        self.reflexion_executor = self.build_reflexion_agent()



    def build_reflexion_prompt(self):
        """Build the Reflexion Multi-Agent Prompt."""
        prompt_hub_template = hub.pull("augustsemrau/reflexion-agent-prompt").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial()
        return prompt



    def build_reflexion_agent(self):
        """Build the Reflexion Multi-Agent System."""
        tool_class = ReflexionTool(course_name=self.course)
        tools = [tool_class.build_reflexion_toolh()]
        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.reflexion_prompt,
                                       output_parser=None)
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                           memory=self.short_term_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        return tas_agent_executor



    def predict(self, query):
        """Invoke the Reflexion Multi-Agent System."""
        print("\n\nUser Query:\n", query)
        response = self.reflexion_executor.invoke({"input": query})[self.output_tag]
        print("\n\nResponse:\n", response)
        # print("\n\nTAS Memory:")
        # print(f"\n{self.tas_executor.memory}\n")
        return response




if __name__ == "__main__":
    student_name = "August"
    student_course = "IntroToMachineLearning"
    student_subject = "Linear Regression"
    # student_learning_preferences = "I prefer formulas and math in order to understand technical concepts"
    student_learning_preferences = "I prefer code examples in order to understand technical concepts"
    # student_learning_preferences = "I prefer text-based explanations and metaphors in order to understand technical concepts"

    student_query = f"Hello, I am {student_name}!\nI am studying the course {student_course} and am trying to learn about the subject {student_subject}.\nMy learning preferences are described as the following: {student_learning_preferences}.\nPlease explain me this subject."


    llm_model = init_llm_langsmith(llm_key=4, temp=0.5, langsmith_name="REFLEXION TEST")
    # llm_model = init_llm_langsmith(llm_key=4, temp=0.5, langsmith_name="BASELINE_CHAIN")
    reflexion = ReflexionMultiAgent(llm_model=llm_model,
                                    baseline_bool=False,
                                    course=student_course,
                                    subject=student_subject,
                                    learning_prefs=student_learning_preferences,
                                    student_name=student_name,
                                    student_id=None#"AugustSemrau1"
                                    )

    res = reflexion.predict(query=student_query)

    res = reflexion.predict(query="I'm not sure I understand the subject from this explanation. Can you explain it in a different way?")

    res = reflexion.predict(query="Thank you for the help, have a nice day!")
