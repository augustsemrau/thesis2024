"""Teaching Agent System (TAS) for the thesis2024 project."""

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
from thesis2024.memory import LongTermMemory
from thesis2024.models.coding_agent import CodingMultiAgent
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



"""Teaching Agent System (TAS) for the thesis2024 project."""
class TAS:
    """Class for the Teaching Agent System."""

    def __init__(self,
                 llm_model,
                 baseline_bool = False,
                 student_id=None,
                 student_name: str = "August",
                 course: str = "IntroToMachineLearning",
                 subject: str = "Linear Regression",
                 learning_prefs: str = "Prefers visual representations of the subject",
                 ):
        """Initialize the Teaching Agent System."""
        self.llm_model = llm_model
        self.course = course
        self.student_id = student_id # If student_id is None, the TAS will not use long-term memory
        self.student_name = student_name

        # Init short term memory for the TAS
        self.short_term_memory = ConversationBufferMemory(memory_key="chat_history",
                                                          return_messages=False,
                                                          ai_prefix="Teaching Assistant",
                                                          human_prefix="Student")

        # Init long term memory for the TAS
        if self.student_id is not None:
            self.long_term_memory_class = LongTermMemory(user_name=self.student_id)

        self.tas_prompt = self.build_tas_prompt(student_name=student_name,
                                                course_name=course,
                                                subject_name=subject,
                                                learning_preferences=learning_prefs,
                                                ltm_query="")
        self.build_executor(ver=baseline_bool)

    """Build the Teaching Agent System executor."""
    def build_executor(self, ver):
        """Build the Teaching Agent System executor."""
        if not ver:
            self.tas_executor = self.build_tas()
            self.output_tag = "output"
        else:
            self.tas_executor = self.build_nonagenic_baseline()
            self.output_tag = "response"

    """Prompt for the Teaching Agent System."""
    def build_tas_prompt(self, student_name, course_name, subject_name, learning_preferences, ltm_query=""):
        """Build the agent prompt."""
        # TODO Advanced memory types such as core_beliefs, formative_events, longterm_memory
        facts = "Nothing"
        if self.student_id is not None:
            facts = self.long_term_memory_class.get_user_semantic_memories(query=ltm_query)
        prompt_hub_template = hub.pull("augustsemrau/react-tas-prompt-2").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial(student_name=student_name,
                                         course_name=course_name,
                                         subject_name=subject_name,
                                         learning_preferences=learning_preferences,
                                         ltm_facts=facts,
                                        )
        # prompt_hub_template = hub.pull("hwchase17/react").template
        # prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        # prompt = prompt_template.partial()
        return prompt

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

    def build_tas(self):
        """Build the Teaching Agent System (TAS).

        This version of the TAS is agenic and has tools.
        """
        tool_class = ToolClass()
        tools = [tool_class.build_search_tool(),
                 tool_class.build_retrieval_tool(course_name=self.course),
                 tool_class.build_coding_tool(),
                #  tool_class.build_math_tool(),
                 ]

        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.tas_prompt,
                                       output_parser=None)#ReActOutputParser())#PydanticOutputParser())#None)#StrOutputParser())
        tas_agent_executor = AgentExecutor(agent=tas_agent,
                                           tools=tools,
                                           memory=self.short_term_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)#"Check your output and make sure it conforms, use the Action/Action Input syntax")#True)
        return tas_agent_executor


    """Predict function for invoking the initiated TAS."""
    def predict(self, query):
        """Invoke the Teaching Agent System."""
        print("\n\nUser Query:", query)
        response = self.tas_executor.invoke({"input": query})[self.output_tag]
        # print("\n\nTAS Memory:")
        # print(f"\n{self.tas_executor.memory}\n")
        if self.student_id is not None:
            self.long_term_memory_class.save_conversation_step(user_query=query, llm_response=response)
        return response


    """(DOES NOT WORK) Predict function for invoking the initiated TAS asynchronously for use in Chainlit frontend."""
    def cl_predict(self, query):
        """Invoke the Teaching Agent System."""

        res = self.tas_executor.invoke({"input": query}, callbacks=[cl.AsyncLangchainCallbackHandler()])
        return res[self.output_tag]










if __name__ == '__main__':

    student_name = "August"
    student_course = "IntroToMachineLearning"
    student_subject = "Linear Regression"
    # student_learning_preferences = "I prefer formulas and math in order to understand technical concepts"
    student_learning_preferences = "I prefer code examples in order to understand technical concepts"
    # student_learning_preferences = "I prefer text-based explanations and metaphors in order to understand technical concepts"

    student_query = f"Hello, I am {student_name}!\nI am studying the course {student_course} and am trying to learn about the subject {student_subject}.\nMy learning preferences are described as the following: {student_learning_preferences}.\nPlease explain me this subject."


    llm_model = init_llm_langsmith(llm_key=4, temp=0.5, langsmith_name="TAS")
    # llm_model = init_llm_langsmith(llm_key=4, temp=0.5, langsmith_name="BASELINE_CHAIN")
    tas = TAS(llm_model=llm_model,
            baseline_bool=False,
            course=student_course,
            subject=student_subject,
            learning_prefs=student_learning_preferences,
            student_name=student_name,
            student_id=None#"AugustSemrau1"
            )

    res = tas.predict(query=student_query)
    print("\n\nResponse:\n", res)
    # res = tas.predict("What is the name of the person who invented the ADAM optimization technique?")
    # print("\n\nResponse: ", res)
    res = tas.predict(query="I'm not sure I understand the subject from this explanation. Can you explain it in a different way?")
    print("\n\nResponse:\n", res)

    res = tas.predict(query="Thank you for the help, have a nice day!")
    print("\n\nResponse:\n", res)




    # Old test queries
    # res = tas.predict("What is the current world record in 50 meter butterfly?")
    # res = tas.predict("Can you explain me how ADAM optimization works?")
    # print(tas.predict("Hej! Jeg vil gerne snakke dansk. Kan du forklare mig hvordan lineær regression virker?"))#["output"]
