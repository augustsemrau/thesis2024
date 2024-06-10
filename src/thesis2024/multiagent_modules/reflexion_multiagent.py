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
        self.learning_preferences = learning_preferences

        # Tools
        self.tool_class = ToolClass()
        self.search_tool = [self.tool_class.build_search_tool()]
        self.retrieval_tool = [self.tool_class.build_retrieval_tool(course_name=self.course)]
        self.crag_tool = [self.tool_class.build_crag_tool(course_name=self.course)]
        self.coding_tool = [self.tool_class.build_coding_tool()]

        self.general_prompt = ""#"You are working together with several other agents in a teaching assistance system."


    def initial_plan(self, query: str, chat_hist: str=""):
        """Create the initial plan."""
        prompt_template = "{system_message}\n\nYou are an expert responsible for creating the initial plan for the response to the student's query:\n{query}\n\nYou must make sure that this plan does not Begin!\n\nThis is the conversation so far, avoid repeating yourself:\n{chat_hist}"
        prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
        prompt = prompt.partial(system_message=self.general_prompt)
        llm_temp = self.llm_model
        chain = prompt | llm_temp | StrOutputParser()
        return chain.invoke(query)


    def crag(self, query: str):#, chat_history: str):#draft: str="", critique: str=""):
        """Build the crag retrieval agent."""
        prompt_hub_template = hub.pull("augustsemrau/crag-agent-prompt-2").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial()#chat_history=chat_history)#draft=draft, critique=critique)
        crag_agent = create_react_agent(llm=self.llm_model,
                                       tools=self.crag_tool,
                                       prompt=prompt)
        executor = AgentExecutor(agent=crag_agent,
                                           tools=self.crag_tool,
                                           verbose=True,
                                           handle_parsing_errors=True)
        retrieved_info = executor.invoke({"input": query})["output"]
        # self.tool_class.run_crag_function(search_query=query, course=self.course)
        return retrieved_info


    def initial_draft(self, query: str, retrieved_info: str, chat_hist: str=""):
        """First draft of the subgraph."""
        prompt_template = "{system_message}\n\nYou are an expert in writing personalized teaching material, and responsible for creating the first draft of the response to the student's query:\n{query}\n\nHere is some external information which may help with improving the draft:\n{retrieved_info}\n\nBegin!\n\nThis is the conversation so far, avoid repeating yourself:\n{chat_hist}"
        prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
        prompt = prompt.partial(system_message=self.general_prompt,
                                retrieved_info=retrieved_info,
                                chat_hist=chat_hist)
        llm_temp = self.llm_model
        chain = prompt | llm_temp | StrOutputParser()
        return chain.invoke(query)


    def new_draft(self, old_draft: str, critique: str, chat_hist: str=""):
        """Create a new draft."""
        prompt_template = "{system_message}\n\nYou are responsible for generating an improved draft given some critique.\n\nThe original draft you must improve is the following:\n{old_draft}\n\nThis draft has been given the following critique:\n{critique}\n\nbased on this critique, improve the draft.\n\nBegin!\n\nThis is the conversation so far, avoid repeating yourself:\n{chat_hist}"
        prompt = PromptTemplate(input_variables=["critique"], template=prompt_template)
        prompt = prompt.partial(system_message=self.general_prompt,
                                old_draft=old_draft,
                                chat_hist=chat_hist)
        llm_temp = self.llm_model
        chain = prompt | llm_temp | StrOutputParser()
        return chain.invoke(critique)




    def critique_draft(self, query: str, draft: str, chat_hist: str=""):
        """Critique the draft."""
        prompt_template = "{system_message}\n\nYou are an expert teaching material supervisor. responsible for critiquing a draft of the response to the student's query:\n\n{query}\n\nYour critique must be MAX 5 sentences.\nThe draft you must critique is the following:\n{draft}\n\nBegin!\n\nThis is the conversation so far:\n{chat_hist}"
        prompt = PromptTemplate(input_variables=["draft"], template=prompt_template)
        prompt = prompt.partial(system_message=self.general_prompt,
                                query=query,
                                chat_hist=chat_hist)
        llm_temp = self.llm_model
        chain = prompt | llm_temp | StrOutputParser()
        return chain.invoke(draft)



    def build_reflexion_tool(self):
        """Build the subgraph tool."""

        def reflexiontool(query: str):
            """Invoke the Reflexion Subgraph."""
            chat_hist = self.short_term_memory.buffer
            plan = self.initial_plan(query=query, chat_hist=chat_hist)
            retrieved_info = ""#self.crag(query=query)
            print(f"Retrieved information:\n{retrieved_info}\n\n")
            draft = self.initial_draft(query=query, retrieved_info=retrieved_info, chat_hist=chat_hist)
            print(f"Initial draft:\n{draft}\n\n")
            for i in range(self.max_iter):
                critique = self.critique_draft(query=query, draft=draft, chat_hist=chat_hist)
                print(f"Critique:\n{critique}\n\n")
                draft = self.new_draft(old_draft=draft, critique=critique, chat_hist=chat_hist)
                print(f"\n\nIteration {i}:\n{draft}\n\n")
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
    student_learning_preferences = "I prefer formulas and math in order to understand technical concepts"
    # student_learning_preferences = "I prefer code examples in order to understand technical concepts"
    # student_learning_preferences = "I prefer text-based explanations and metaphors in order to understand technical concepts"

    student_query = f"Hello, I am {student_name}!\nI am studying the course {student_course} and am trying to learn about the subject {student_subject}.\nMy learning preferences are described as the following: {student_learning_preferences}.\nPlease explain me this subject."

    llm_model = init_llm_langsmith(llm_key=3, temp=0.2, langsmith_name="REFLEXION TEST")
    reflexion = ReflexionMultiAgent(llm_model=llm_model,
                                    max_iter=2,
                                    course=student_course,
                                    learning_preferences=student_learning_preferences,
                                    )

    res = reflexion.predict(query=student_query)
    res = reflexion.predict(query="I'm not sure I understand the subject from this explanation. Can you explain it in a different way?")
    res = reflexion.predict(query="Thank you for the help, have a nice day!")
