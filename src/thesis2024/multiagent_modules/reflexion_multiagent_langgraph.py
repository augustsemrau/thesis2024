from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_core.pydantic_v1 import BaseModel

from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

# Local imports
from thesis2024.utils import init_llm_langsmith
from thesis2024.tools import ToolClass

from tavily import TavilyClient
import os


memory = SqliteSaver.from_conn_string(":memory:")


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

class Queries(BaseModel):
    queries: List[str]



class ReflexionMultiAgentLG:
    def __init__(self,
                 llm_model,
                 max_iter: int = 3,
                 course: str = "IntroToMachineLearning",
                 ):
        """Initialize the Teaching Agent System."""
        self.llm_model = llm_model
        self.short_term_memory = ConversationBufferMemory(memory_key="chat_history",
                                                          return_messages=False,
                                                          ai_prefix="Teaching Assistant",
                                                          human_prefix="Student")

        self.max_iter = max_iter
        self.course = course

        # Tools
        self.tool_class = ToolClass()
        self.search_tool = [self.tool_class.build_search_tool()]
        self.retrieval_tool = [self.tool_class.build_retrieval_tool(course_name=self.course)]
        self.crag_tool = [self.tool_class.build_crag_tool()]
        self.coding_tool = [self.tool_class.build_coding_tool()]

        self.general_prompt = "You are working together with several other agents in a teaching assistance system.\n\n"
        self.tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        self.graph = self.build_graph()


    def plan_node(self, state: AgentState):
        """Plan Node."""
        plan_prompt = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""
        messages = [SystemMessage(content=plan_prompt), HumanMessage(content=state['task'])]
        response = self.llm_model.invoke(messages)
        return {"plan": response.content}


    def research_plan_node(self, state: AgentState):
        """Research Plan Node."""
        plan_research_prompt = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""
        queries = self.llm_model.with_structured_output(Queries).invoke([
            SystemMessage(content=plan_research_prompt),
            HumanMessage(content=state['task'])
        ])
        content = state['content'] or []
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}


    def draft_node(self, state: AgentState):
        """Drafting Node."""
        draft_prompt = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed:

------

{content}"""
        content = "\n\n".join(state['content'] or [])
        user_message = HumanMessage(content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
        messages = [SystemMessage(content=draft_prompt.format(content=content)), user_message]
        response = self.llm_model.invoke(messages)
        return {"draft": response.content, "revision_number": state.get("revision_number", 1) + 1}


    def reflection_node(self, state: AgentState):
        """Reflection Node."""
        reflection_prompt = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""
        messages = [
            SystemMessage(content=reflection_prompt),
            HumanMessage(content=state['draft'])
        ]
        response = self.llm_model.invoke(messages)
        return {"critique": response.content}


    def research_critique_node(self, state: AgentState):
        """Research Critique Node."""
        critique_research_prompt = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""
        queries = self.llm_model.with_structured_output(Queries).invoke([
            SystemMessage(content=critique_research_prompt),
            HumanMessage(content=state['critique'])
        ])
        content = state['content'] or []
        for q in queries.queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}



    def should_continue(self, state):
        """Check if the agent should continue based on iteration count."""
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"






    def build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("generate", self.draft_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("research_critique", self.research_critique_node)

        builder.set_entry_point("planner")
        builder.add_conditional_edges(
            "generate",
            self.should_continue,
            {END: END, "reflect": "reflect"}
        )
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")

        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")

        graph = builder.compile(checkpointer=self.short_term_memory)
        return graph



    def predict(self, query: str):
        """Invoke the Teaching Multi-Agent System."""
        thread = {"configurable": {"thread_id": "1"}}
        outputs = []
        for s in self.graph.stream({
            'task': query,
            "max_revisions": self.max_iter,
            "revision_number": 1,
        }, thread):
            print(f"\n\nstep: {s}")
            outputs.append(s)
        return outputs[-1]["generate"]["draft"]




if __name__ == "__main__":
    student_name = "August"
    student_course = "IntroToMachineLearning"
    student_subject = "Linear Regression"
    student_learning_preferences = "I prefer formulas and math in order to understand technical concepts"
    # student_learning_preferences = "I prefer code examples in order to understand technical concepts"
    # student_learning_preferences = "I prefer text-based explanations and metaphors in order to understand technical concepts"

    student_query = f"Hello, I am {student_name}!\nI am studying the course {student_course} and am trying to learn about the subject {student_subject}.\nMy learning preferences are described as the following: {student_learning_preferences}.\nPlease explain me this subject."

    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="REFLEXION TEST")
    reflexion = ReflexionMultiAgentLG(llm_model=llm_model,
                                    max_iter=2,
                                    course=student_course,
                                    )

    res = reflexion.predict(query=student_query)
    res = reflexion.predict(query="I'm not sure I understand the subject from this explanation. Can you explain it in a different way?")
    print(res)
    # res = reflexion.predict(query="Thank you for the help, have a nice day!")
