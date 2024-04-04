"""This script is esentially the entire Langgraph multi-agent-colaboration example 
put into a single class.
Currently, it will only print it's output, and not return it to the main script."""

# LangChain imports
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain




import json

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation





from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults






import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union

from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from typing_extensions import TypedDict
import functools

# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str







class CodingMultiAgent:
    """Multi-Agent coding LangGraph model."""

    def __init__(self, llm: str="gpt-3.5-turbo-0125"):
        """Initialize the MultiAgent class."""
        self.llm = llm
        return None

    # Helper functions
    def create_agent(self, tools, system_message):
        """Create arbitrary agent."""
        functions = [format_tool_to_openai_function(t) for t in tools]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        return prompt | self.llm.bind_functions(functions)

    def agent_node(self, state, agent, name):
        """Helper function to create a node for a given agent. Node that invokes agent."""
        result = agent.invoke(state)
        # We convert the agent output into a format that is suitable to append to the global state
        if isinstance(result, FunctionMessage):
            pass
        else:
            result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            # Since we have a strict workflow, we can
            # track the sender so we know who to pass to next.
            "sender": name,
        }


    # Tool setup functions
    def setup_tavily_tool(self):
        """Setups Tavily Search Tool."""
        tavily_tool = TavilySearchResults(max_results=5)
        return tavily_tool

    def setup_rag_tool(self):
        """Setups RAG Tool."""
        rag_tool = load_qa_chain("facebook/rag-token-nq")
        return rag_tool

    def setup_coding_tool(self):
        """Setups Coding Tool."""
        # Warning: This executes code locally, which can be unsafe when not sandboxed
        repl = PythonREPL()

        @tool
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
        return python_repl


    # Agent and Node creation functions
    def create_research_node(self):
        """Create research agent and node."""
        research_agent_tools =[self.setup_tavily_tool()]
        research_agent = self.create_agent(research_agent_tools,
            system_message="You should provide accurate data for the code generator to use.",
        )
        research_node = functools.partial(self.agent_node, agent=research_agent, name="Researcher")
        return research_node

    def create_code_node(self):
        """Create code agent and node."""
        code_agent = self.create_agent([self.setup_coding_tool()],
            system_message="Any code you execute will be visible by the user. If you successfully execute the desired code, prefix your response with FINAL ANSWER.",
        )
        code_node = functools.partial(self.agent_node, agent=code_agent, name="Code Generator")
        return code_node

    def create_tool_node(self, state):
        """Create tool node for running tools in the graph.

        It takes in an agent action and calls that tool and returns the result.
        """
        tavily_tool = self.setup_tavily_tool()
        python_repl = self.setup_coding_tool()
        tools = [tavily_tool, python_repl]
        tool_executor = ToolExecutor(tools)



        messages = state["messages"]
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an ToolInvocation from the function_call
        tool_input = json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        )
        # We can pass single-arg inputs by value
        if len(tool_input) == 1 and "__arg1" in tool_input:
            tool_input = next(iter(tool_input.values()))
        tool_name = last_message.additional_kwargs["function_call"]["name"]
        action = ToolInvocation(
            tool=tool_name,
            tool_input=tool_input,
        )
        # We call the tool_executor and get back a response
        response = tool_executor.invoke(action)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(
            content=f"{tool_name} response: {str(response)}", name=action.tool
        )
        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}


    # Edge function
    def router_edge(self, state):
        """Router for the graph.

        This function decides which node to go to next based on the state.
        """
        # This is the router
        messages = state["messages"]
        last_message = messages[-1]
        if "function_call" in last_message.additional_kwargs:
            # The previus agent is invoking a tool
            return "call_tool"
        if "FINAL ANSWER" in last_message.content:
            # Any agent decided the work is done
            return "end"
        return "continue"


    # Graph setup
    def instanciate_graph(self):
        """Instantiate the graph."""
        # Nodes must be created before the graph is instantiated
        research_node = self.create_research_node()
        code_node = self.create_code_node()

        # We create a graph and add the nodes
        workflow = StateGraph(AgentState)

        workflow.add_node("Researcher", research_node)
        workflow.add_node("Code Generator", code_node)
        workflow.add_node("call_tool", self.create_tool_node)

        workflow.add_conditional_edges(
            "Researcher",
            self.router_edge,
            {"continue": "Code Generator", "call_tool": "call_tool", "end": END},
        )
        workflow.add_conditional_edges(
            "Code Generator",
            self.router_edge,
            {"continue": "Researcher", "call_tool": "call_tool", "end": END},
        )

        workflow.add_conditional_edges(
            "call_tool",
            # Each agent node updates the 'sender' field
            # the tool calling node does not, meaning
            # this edge will route back to the original agent
            # who invoked the tool
            lambda x: x["sender"],
            {
                "Researcher": "Researcher",
                "Code Generator": "Code Generator",
            },
        )
        workflow.set_entry_point("Researcher")
        graph = workflow.compile()
        return graph


if __name__ == "__main__":

    coding_class = CodingMultiAgent(llm="gpt-3.5-turbo-0125")

    coding_graph = coding_class.instanciate_graph()
    print(coding_graph)


