"""Teaching Agent System (TAS) for the thesis2024 project."""

# Langchain imports
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Tool imports
from langchain.tools import StructuredTool

# Local imports
from thesis2024.utils import init_llm_langsmith
from thesis2024.tools import ToolClass
from thesis2024.PMAS import LongTermMemory


from thesis2024.multiagent_modules.coding_agent import CodingMultiAgent
from thesis2024.multiagent_modules.reflexion_multiagent import ReflexionMultiAgent
from thesis2024.multiagent_modules.hierarchical_multiagent import HierarchicalMultiAgent




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




"""Teaching Multi-Agent System (TMAS) for the thesis2024 project."""
class TMAS:
    """Class for the TMAS."""

    def __init__(self,
                 llm_model,
                 reflexion_bool = False,
                 reflexion_iters: int = 3,
                 course: str = "IntroToMachineLearning",
                 student_id=None,
                 ):
        """Initialize the TMAS."""
        self.llm_model = llm_model
        self.course = course

        # Init long term memory for the TAS
        self.student_id = student_id # If student_id is None, the TAS will not use long-term memory
        if self.student_id is not None:
            self.long_term_memory_class = LongTermMemory(user_name=self.student_id)

        """Build the TMAS Class."""
        if reflexion_bool:
            self.tmas_class = ReflexionMultiAgent(llm_model=llm_model,
                                                max_iter=reflexion_iters,
                                                course=course,)
        else:
            self.tmas_class = HierarchicalMultiAgent()


    def predict(self, query):
        """Invoke the Teaching Multi-Agent System."""
        print("\n\nUser Query:", query)
        response = self.tmas_class.predict(query=student_query)
        print("\n\nResponse:\n", res)
        if self.student_id is not None:
            self.long_term_memory_class.save_conversation_step(user_query=query, llm_response=response)
        return response







if __name__ == '__main__':

    student_name = "August"
    student_course = "IntroToMachineLearning"
    student_subject = "Linear Regression"
    # student_learning_preferences = "I prefer formulas and math in order to understand technical concepts"
    student_learning_preferences = "I prefer code examples in order to understand technical concepts"
    # student_learning_preferences = "I prefer text-based explanations and metaphors in order to understand technical concepts"

    student_query = f"Hello, I am {student_name}!\nI am studying the course {student_course} and am trying to learn about the subject {student_subject}.\nMy learning preferences are described as the following: {student_learning_preferences}.\nPlease explain me this subject."

    reflexion=True
    if reflexion:
        llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="REFLEXION TMAS Testing")
    else:
        llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="HIERARCHICAL TMAS Testing")

    tmas = TMAS(llm_model=llm_model,
            reflexion_bool=reflexion,
            reflexion_iters=3,
            course=student_course,
            student_id=None#"AugustSemrau1"
            )

    res = tmas.predict(query=student_query)
    res = tmas.predict(query="I'm not sure I understand the subject from this explanation. Can you explain it in a different way?")
    res = tmas.predict(query="Thank you for the help, have a nice day!")
