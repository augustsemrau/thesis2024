"""Mmodule containings the reflexion agent framework."""

# Langchain imports
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Tool imports
from langchain.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated
from langchain_experimental.utilities import PythonREPL

# Local imports
from thesis2024.utils import init_llm_langsmith
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
        self.search_tool = [tool_class.build_search_tool()]
        self.retrieval_tool = [tool_class.build_retrieval_tool(course_name=self.course)]
        self.coding_tool = [tool_class.build_coding_tool()]

        self.general_prompt = "You are working together with several other agents in a teaching assistance system.\n\n"

    def initial_draft(self, query: str, chat_history: str):
        """First draft of the subgraph."""
        prompt_template = "{system_message}\n\nYou are responsible for creating the first draft of the response to the student's query:\n\n{query}\n\nThis is the previous interactions between the teaching agent system and the student: {chat_history}\n\nBegin!"
        prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
        prompt = prompt.partial(system_message=self.general_prompt,
                                chat_history=chat_history)
        llm_temp = self.llm_model
        chain = prompt | llm_temp | StrOutputParser()
        return chain.invoke(query)

    def critique_draft(self, query: str, draft: str):
        """Critique the draft."""
        prompt_template = "{system_message}\n\nYou are responsible for critiquing a draft of the response to the student's query:\n\n{query}\n\nThe draft you must critique is the following:\n{draft}\n\nBegin!"
        prompt = PromptTemplate(input_variables=["draft"], template=prompt_template)
        prompt = prompt.partial(system_message=self.general_prompt,
                                query=query)
        llm_temp = self.llm_model
        chain = prompt | llm_temp | StrOutputParser()
        return chain.invoke(draft)

    def new_draft(self, old_draft: str, critique: str):
        """Create a new draft."""
        prompt_template = "{system_message}\n\nYou are responsible for generating an improved draft given some critique.\n\nThe original draft you must improve is the following:\n{old_draft}\n\nThis draft has been given the following critique:\n{critique}\n\nbased on this critique, improve the draft.\n\nBegin!"
        prompt = PromptTemplate(input_variables=["critique"], template=prompt_template)
        prompt = prompt.partial(system_message=self.general_prompt,
                                old_draft=old_draft)
        llm_temp = self.llm_model
        chain = prompt | llm_temp | StrOutputParser()
        return chain.invoke(critique)


    def build_reflexion_tool(self):
        """Build the subgraph tool."""

        def reflexiontool(query: str, chat_history: str = ""):
            """Invoke the Reflexion Subgraph."""
            draft = self.initial_draft(query, chat_history)
            print(f"Initial draft:\n{draft}\n\n")
            for i in range(self.max_iter):
                critique = self.critique_draft(query=query, draft=draft)
                # print(f"Critique:\n{critique}\n\n")
                draft = self.new_draft(old_draft=draft, critique=critique)
                # print(f"\n\nIteration {i}:\n{draft}\n\n")
            response = draft
            return response

        reflexion_tool = StructuredTool.from_function(
                            name="Reflexion Tool",
                            func=reflexiontool,
                            description="Useful when you need to answer questions.",
                            return_direct=True
                            )
        return reflexion_tool

    def build_finishconversation_tool(self):
        """Build the finish conversation tool."""
        def finishconversation(query: str):
            return "Thank you for the help, have a nice day!"

        end_finishconversation_tool = StructuredTool.from_function(
                            name="Finish Conversation Tool",
                            func=finishconversation,
                            description="Useful when you need to end a conversation.",
                            return_direct=True
                            )
        return end_finishconversation_tool





"""Teaching Agent System (TAS) for the thesis2024 project."""
class ReflexionMultiAgent:
    """Class for the Teaching Agent System."""

    def __init__(self,
                 llm_model,
                 reflexion_model,
                 max_iter: int = 3,
                 course: str = "IntroToMachineLearning",
                 ):
        """Initialize the Teaching Agent System."""
        self.llm_model = llm_model
        self.reflexion_model = reflexion_model
        self.short_term_memory = ConversationBufferMemory(memory_key="chat_history",
                                                          return_messages=False,
                                                          ai_prefix="Teaching Assistant",
                                                          human_prefix="Student")
        self.reflexion_class = ReflexionTool(llm_model=self.reflexion_model,
                                             max_iterations=max_iter,
                                             course_name=course)
        self.reflexion_executor = self.build_executor()


    def build_executor(self):
        """Build the Reflexion Multi-Agent System."""
        reflexion_tool = [self.reflexion_class.build_reflexion_tool(),
                          self.reflexion_class.build_finishconversation_tool()]
        prompt_hub_template = hub.pull("augustsemrau/reflexion-agent-prompt").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial()
        tas_agent = create_react_agent(llm=self.llm_model,
                                       tools=reflexion_tool,
                                       prompt=prompt,
                                       output_parser=None)
        executor = AgentExecutor(agent=tas_agent,
                                           tools=reflexion_tool,
                                           memory=self.short_term_memory,
                                           verbose=False,
                                           handle_parsing_errors=True)
        return executor

    def predict(self, query):
        """Invoke the Reflexion Multi-Agent System."""
        print("\n\nUser Query:\n", query)
        response = self.reflexion_executor.invoke({"input": query})["output"]
        print("\n\nResponse:\n", response)
        # print(f"\n\nTAS Memory:\n{self.tas_executor.memory}\n")
        return response




if __name__ == "__main__":
    student_name = "August"
    student_course = "IntroToMachineLearning"
    student_subject = "Linear Regression"
    # student_learning_preferences = "I prefer formulas and math in order to understand technical concepts"
    # student_learning_preferences = "I prefer code examples in order to understand technical concepts"
    student_learning_preferences = "I prefer text-based explanations and metaphors in order to understand technical concepts"

    student_query = f"Hello, I am {student_name}!\nI am studying the course {student_course} and am trying to learn about the subject {student_subject}.\nMy learning preferences are described as the following: {student_learning_preferences}.\nPlease explain me this subject."


    reflexion_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="REFLEXION TEST")
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="REFLEXION TEST")
    # llm_model = init_llm_langsmith(llm_key=4, temp=0.5, langsmith_name="BASELINE_CHAIN")
    reflexion = ReflexionMultiAgent(llm_model=llm_model,
                                    reflexion_model=reflexion_model,
                                    max_iter=1,
                                    course=student_course,
                                    )

    res = reflexion.predict(query=student_query)
    res = reflexion.predict(query="I'm not sure I understand the subject from this explanation. Can you explain it in a different way?")
    res = reflexion.predict(query="Thank you for the help, have a nice day!")
