"""Teaching Agent System (TAS) for the thesis2024 project."""

# Langchain imports
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
import chainlit as cl

# Local imports
from thesis2024.utils import init_llm_langsmith
from thesis2024.PMAS import LongTermMemory
from thesis2024.tools import ToolClass



class TAS:
    """Teaching Agent System (TAS) for the thesis2024 project."""

    def __init__(self,
                 llm_model,
                 baseline_bool = False,
                 student_name: str = "August",
                 course: str = "IntroToMachineLearning",
                 subject: str = "Linear Regression",
                 learning_prefs: str = "Prefers visual representations of the subject",
                 student_id=None,
                 ltm_query="",
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
                                                ltm_query=ltm_query)
        self.build_executor(ver=baseline_bool)

    def build_executor(self, ver):
        """Build the Teaching Agent System executor."""
        if not ver:
            self.tas_executor = self.build_tas()
            self.output_tag = "output"
        else:
            self.tas_executor = self.build_nonagenic_baseline()
            self.output_tag = "response"

    def build_tas_prompt(self, student_name, course_name, subject_name, learning_preferences, ltm_query):
        """Build the agent prompt."""
        facts = "No prior conversations for this student."
        if self.student_id is not None and ltm_query != "":
            facts = self.long_term_memory_class.predict(query=ltm_query)
        prompt_hub_template = hub.pull("augustsemrau/react-tas-prompt-3").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial(student_name=student_name,
                                         course_name=course_name,
                                         subject_name=subject_name,
                                         learning_preferences=learning_preferences,
                                         ltm_facts=facts,
                                        )
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
        baseline_chain = ConversationChain(llm=self.llm_model,
                                prompt=prompt,
                                memory=self.short_term_memory,
                                #output_parser=BaseLLMOutputParser(),
                                verbose=False,)
        return baseline_chain

    def build_tas(self):
        """Build the Teaching Agent System (TAS)."""
        tool_class = ToolClass()
        tools = [tool_class.build_search_tool(),
                 tool_class.build_retrieval_tool(course_name=self.course),
                #  tool_class.build_coding_tool(),
                ]

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


    def predict(self, query):
        """Invoke the Teaching Agent System."""
        print("\n\nUser Query:\n", query)
        response = self.tas_executor.invoke({"input": query})[self.output_tag]
        print("\n\nResponse:\n", response)

        if self.student_id is not None:
            self.long_term_memory_class.save_conversation_step(user_query=query, llm_response=response)
        return response

    def save_conversation_step(self, user_query, llm_response):
        """Save the conversation step in the long-term memory."""
        if self.student_id is not None:
            self.long_term_memory_class.save_conversation_step(user_query=user_query, llm_response=llm_response)
        return None


    """TODO (DOES NOT WORK) Predict function for invoking the initiated TAS asynchronously for use in Chainlit frontend."""
    def cl_predict(self, query):
        """Invoke the Teaching Agent System."""
        res = self.tas_executor.invoke({"input": query}, callbacks=[cl.AsyncLangchainCallbackHandler()])
        return res[self.output_tag]










if __name__ == '__main__':

    student_name = "August"
    student_course = "IntroToMachineLearning"
    student_subject = "Linear Regression"
    student_learning_preferences = "I prefer formulas and math in order to understand technical concepts"
    # student_learning_preferences = "I prefer code examples in order to understand technical concepts"
    # student_learning_preferences = "I prefer text-based explanations and metaphors in order to understand technical concepts"

    student_query = f"Hello, I am {student_name}!\nI am studying the course {student_course} and am trying to learn about the subject {student_subject}.\nMy learning preferences are described as the following: {student_learning_preferences}.\nPlease explain me this subject."

    llm_model = init_llm_langsmith(llm_key=4, temp=0.5, langsmith_name="TAS Few-Shot-Examples")
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
    res = tas.predict(query="I'm not sure I understand the subject from this explanation. Can you explain it in a different way?")
    res = tas.predict(query="Thank you for the help, have a nice day!")


