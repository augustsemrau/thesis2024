"""Module containing the simulated student agent system."""

import os
import getpass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.adapters.openai import convert_message_to_dict

from langgraph.graph import END, MessageGraph

from typing import List

import openai

def init_llm_langsmith(llm_key = 3):
    """Initialize the LLM model for the LangSmith system."""
    # Set environment variables
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass(f"Please provide your {var}")
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_API_KEY")
    # Add tracing in LangSmith.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if llm_key == 3:
        llm = "gpt-3.5-turbo-0125"
        os.environ["LANGCHAIN_PROJECT"] = "GPT-3.5 Simulated Student Agent"
    elif llm_key == 4:
        llm = "gpt-4-0125-preview"
        os.environ["LANGCHAIN_PROJECT"] = "GPT-4 Simulated Student Agent"
    return llm




class SimulatedStudentAgent:
    """Class for the simulated student agent system."""

    def __init__(self, llm: str="gpt-3.5-turbo-0125",
                ):
        """Initialize the Assessor Agent class."""
        self.model = ChatOpenAI(model_name=llm, temperature=0.5)
        return None


    def create_simulated_student_agent_chain(self):
        """Create the simulated student agent chain."""
        system_prompt_template = """You are a university student in STEM. \
        You are interacting with a user who is a teaching assistant for a given subject. \

        {instructions}

        When you are finished with the conversation, respond with a single word 'FINISHED'"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        instructions = """Your name is Felix. \
                        You want to learn Deep Learning and in particular the subject of gradient descent. \
                        You have only fundamental knowledge of math, statistics, and programming. \
                        Your goal is to understand the basics of Deep Learning. \
                        You will never write more than 3 sentences at a time, and all you write is not correct. """
        prompt = prompt.partial(name="Felix", instructions=instructions)
        model = ChatOpenAI()
        simulated_student_agent_chain = prompt | model
        return simulated_student_agent_chain




# This is flexible, but you can define your agent here, or call your agent API here.
def my_chat_bot(messages: List[dict]) -> dict:
    system_message = {
        "role": "system",
        "content": """You are a STEM teaching assistant. \
                    Your goal is firstly to explain subjects of interest. \
                    Secondly, you should probe the students understanding of the subject by \
                    asking questions that can help a bystanding teacher assess how well the student is doing. \
                        The relevant subject is Deep Learning and the topic is gradient descent. """,
    }
    messages = [system_message] + messages
    completion = openai.chat.completions.create(
        messages=messages, model="gpt-3.5-turbo"
    )
    return completion.choices[0].message.model_dump()

def chat_bot_node(messages):
    # Convert from LangChain format to the OpenAI format, which our chatbot function expects.
    messages = [convert_message_to_dict(m) for m in messages]
    # Call the chat bot
    chat_bot_response = my_chat_bot(messages)
    # Respond with an AI Message
    return AIMessage(content=chat_bot_response["content"])

def _swap_roles(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return new_messages


def should_continue(messages):
    if len(messages) > 6:
        return "end"
    elif messages[-1].content == "FINISHED":
        return "end"
    else:
        return "continue"



if __name__ == "__main__":

    llm_ver = init_llm_langsmith(llm_key=3)

    simulated_student_agent_class = SimulatedStudentAgent(llm=llm_ver)
    simulated_student_agent_chain = simulated_student_agent_class.create_simulated_student_agent_chain()

    # output = simulated_student_agent_chain.invoke({"messages": [HumanMessage(content="Hi! How can I help you?")]})
    # print(output)

    def simulated_user_node(messages):
        # Swap roles of messages
        new_messages = _swap_roles(messages)
        # Call the simulated user
        response = simulated_student_agent_chain.invoke({"messages": new_messages})
        # This response is an AI message - we need to flip this to be a human message
        return HumanMessage(content=response.content)


    my_chat_bot([{"role": "user", "content": "hi!"}])

    graph_builder = MessageGraph()
    graph_builder.add_node("user", simulated_user_node)
    graph_builder.add_node("chat_bot", chat_bot_node)
    # Every response from  your chat bot will automatically go to the
    # simulated user
    graph_builder.add_edge("chat_bot", "user")
    graph_builder.add_conditional_edges(
        "user",
        should_continue,
        # If the finish criteria are met, we will stop the simulation,
        # otherwise, the virtual user's message will be sent to your chat bot
        {
            "end": END,
            "continue": "chat_bot",
        },
    )
    # The input will first go to your chat bot
    graph_builder.set_entry_point("chat_bot")
    simulation = graph_builder.compile()

    for chunk in simulation.stream([]):
        # Print out all events aside from the final end chunk
        if END not in chunk:
            print(chunk)
            print("----")

