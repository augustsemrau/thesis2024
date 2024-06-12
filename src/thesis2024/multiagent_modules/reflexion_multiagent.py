"""Mmodule containings the reflexion agent framework."""

# Langchain imports
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Tool imports
from langchain.tools import StructuredTool

# Local imports
from thesis2024.utils import init_llm_langsmith
from thesis2024.tools import ToolClass






"""Teaching Agent System (TAS) for the thesis2024 project."""
class ReflexionMultiAgent:
    """Class for the Teaching Agent System."""

    def __init__(self,
                 llm_model,
                 max_iter: int = 3,
                 course: str = "IntroToMachineLearning",
                 subject: str = "Linear Regression",
                 learning_preferences: str = "",
                 ):
        """Initialize the Teaching Agent System."""
        self.llm_model = llm_model
        self.short_term_memory = ConversationBufferMemory(memory_key="chat_history",
                                                          return_messages=False,
                                                          ai_prefix="Teaching Assistant",
                                                          human_prefix="Student")
        self.reflexion_executor = self.build_executor()

        self.max_iter = max_iter
        self.course = course
        self.subject = subject
        self.learning_preferences = learning_preferences

        # Tools
        self.tool_class = ToolClass()
        self.search_tool = [self.tool_class.build_search_tool()]
        self.retrieval_tool = [self.tool_class.build_retrieval_tool(course_name=self.course)]
        self.crag_tool = [self.tool_class.build_crag_tool(course_name=self.course)]
        self.coding_tool = [self.tool_class.build_coding_tool()]

    def plan(self, query: str, chat_hist: str=""):
        """Create the initial plan."""
        prompt_template = """You are an expert teacher, tasked with writing an outline for personalized teaching material.
The subject is {subject} from the course {course}.
The student has the following learning preferences, which must be adhered to:
{learning_preferences}.
\nMake sure that the answer does not repeat the same content as previous answers, if any:\n{chat_hist}
\n\nNow, write a high level outline for answering the following question:\n{query}
\nBegin!
"""
        prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
        prompt = prompt.partial(subject=self.subject,
                                course=self.course,
                                learning_preferences=self.learning_preferences,
                                chat_hist=chat_hist)
        chain = prompt | self.llm_model | StrOutputParser()
        return chain.invoke(query)


    def research_plan(self, outline: str):
        """Build the crag retrieval agent."""
        prompt_hub_template = hub.pull("augustsemrau/crag-agent-prompt-3").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial()
        crag_agent = create_react_agent(llm=self.llm_model,
                                       tools=self.crag_tool,
                                       prompt=prompt)
        executor = AgentExecutor(agent=crag_agent,
                                           tools=self.crag_tool,
                                           verbose=False,
                                           handle_parsing_errors=True)
        retrieved_info = executor.invoke({"input": outline})["output"]
        return retrieved_info


    def initial_draft(self, outline: str, retrieved_info: str):
        """First draft of the subgraph."""
        prompt_template = """You are an expert teacher tasked with writing personalized teaching material.
Here is the student's learning preferences, which must be adhered to:
{learning_preferences}
\n\nThis information may help you:
{retrieved_info}
\n\nGenerate the best personalized teaching material possible.
The material should be coherent and easy to understand for the student.
Use the following outline as a starting point for what topics to include, and the order to write them in.
{outline}
"""
        prompt = PromptTemplate(input_variables=["outline"], template=prompt_template)
        prompt = prompt.partial(learning_preferences=self.learning_preferences,
                                retrieved_info=retrieved_info)
        chain = prompt | self.llm_model | StrOutputParser()
        return chain.invoke(outline)

    def critique_draft(self, query: str, draft: str, chat_hist: str=""):
        """Critique the draft."""
        prompt_template = """You are an expert teaching, critiquing personalized teaching material.
The material is an answer to the following query:
{query}
\nThese are the learning preferences of the student:\n{learning_preferences}
\nIt is very important that the material does not repeat the same content as previous interactions, if there have been any.
These are the interactions so far:
{chat_hist}
\n\nYour critique must be MAX 5 sentences.
The personalized teaching material you must critique is the following:
{draft}
"""
        prompt = PromptTemplate(input_variables=["draft"], template=prompt_template)
        prompt = prompt.partial(query=query,
                                learning_preferences=self.learning_preferences,
                                chat_hist=chat_hist)
        chain = prompt | self.llm_model | StrOutputParser()
        return chain.invoke(draft)


    def new_draft(self, old_draft: str, critique: str, chat_hist: str=""):
        """Create a new draft."""
        prompt_template = """You are an expert teacher, responsible for improving personalized teaching material which has been critiqued.
The original material you must improve in accordance to the critique is the following:
{old_draft}
\nDO NOT incorporate the critique itself into the new material.
Based on the following critique, improve the personalized teaching material:
{critique}
"""
        prompt = PromptTemplate(input_variables=["critique"], template=prompt_template)
        prompt = prompt.partial(old_draft=old_draft,
                                chat_hist=chat_hist)
        llm_temp = self.llm_model
        chain = prompt | llm_temp | StrOutputParser()
        return chain.invoke(critique)


    def final_draft(self, draft: str):
        """Create the final draft."""
        prompt_template = """You are an expert writer.
The following personalized teaching material has been critiqued and improved, but some of the critique may have been left inside.
\nFinalize the material by rewriting it to be a coherent piece of text, suitable for a student to read.
You should remove any unnecessary information which is not relevant to the student, but keep all points, the scope and length of the material.
Never remove mathematical formulas or code examples, as they are crucial for the student's understanding.
\n\nFinalize the following personalized teaching material:
{draft}
"""
        prompt = PromptTemplate(input_variables=["draft"], template=prompt_template)
        chain = prompt | self.llm_model | StrOutputParser()
        return chain.invoke(draft)



    def build_reflexion_tool(self):
        """Build the subgraph tool."""
        def reflexiontool(query: str):
            """Invoke the Reflexion Subgraph."""
            chat_hist = self.short_term_memory.buffer
            outline = self.plan(query=query, chat_hist=chat_hist)
            print(f"Outline:\n{outline}\n\n")
            retrieved_info = self.research_plan(outline=outline)
            # print(f"Retrieved information:\n{retrieved_info}\n\n")
            draft = self.initial_draft(outline=outline, retrieved_info=retrieved_info)
            # print(f"Initial draft:\n{draft}\n\n")
            for i in range(self.max_iter):
                critique = self.critique_draft(query=query, draft=draft, chat_hist=chat_hist)
                # print(f"Critique:\n{critique}\n\n")
                draft = self.new_draft(old_draft=draft, critique=critique, chat_hist=chat_hist)
                # print(f"\n\nIteration {i}:\n{draft}\n\n")
            draft = self.final_draft(draft=draft)
            return draft

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
            return "No problem, have a nice day!"

        end_finishconversation_tool = StructuredTool.from_function(
                            name="Finish Conversation Tool",
                            func=finishconversation,
                            description="Useful when you need to end a conversation.",
                            return_direct=True
                            )
        return end_finishconversation_tool

    def build_executor(self):
        """Build the Reflexion Multi-Agent System."""
        reflexion_tool = [self.build_reflexion_tool(),
                          self.build_finishconversation_tool()]
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
        return response




if __name__ == "__main__":
    student_name = "August"
    student_course = "IntroToMachineLearning"
    student_subject = "Linear Regression"
    # student_learning_preferences = "I prefer formulas and math in order to understand technical concepts"
    student_learning_preferences = "I prefer code examples in order to understand technical concepts"
    # student_learning_preferences = "I prefer text-based explanations and metaphors in order to understand technical concepts"

    student_query = f"Hello, I am {student_name}!\nI am studying the course {student_course} and am trying to learn about the subject {student_subject}.\nMy learning preferences are described as the following: {student_learning_preferences}.\nPlease explain me this subject."

    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="REFLEXION TEST")
    reflexion = ReflexionMultiAgent(llm_model=llm_model,
                                    max_iter=2,
                                    course=student_course,
                                    subject=student_subject,
                                    learning_preferences=student_learning_preferences,
                                    )

    res = reflexion.predict(query=student_query)
    res = reflexion.predict(query="I'm not sure I understand the subject from this explanation. Can you explain it in a different way?")
    res = reflexion.predict(query="Thank you for the help, have a nice day!")
