"""Module containing the simulated student agent (SSA) system."""

# LangChain imports
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Local Imports
from thesis2024.utils import init_llm_langsmith




class SSA:
    """Class for the simulated student agent system."""

    def __init__(self,
                 llm_model,
                 version: str = "v0"):
        """Initialize the Assessor Agent class."""
        self.llm_model = llm_model
        self.ssa_prompt = self.build_ssa_prompt()

        if version == "v0":
            self.ssa_executor = self.build_ssa_v0()

    def build_ssa_prompt(self):
        """Build the agent prompt."""
        system_message = """Your name is Felix, the student. 
You want to learn and you are curios. 
Your goal is to understand the basics and be able to reflect on the given topic. 
You will never write more than 3 sentences at a time.
The last message you write should be 'FINISHED' to end the conversation."""
        course = """Introduction to Computer Science"""
        subject = """Gradient Descent"""
        prior_knowledge = """You have a fundamental knowledge of math, statistics, and programming."""
        prior_knowledge = """You are clueless about anything related to the subject."""

        prompt_hub_template = hub.pull("augustsemrau/simulated-student-agent").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial(system_message=system_message,
                                         course_name=course,
                                         subject_name=subject,
                                         prior_knowledge=prior_knowledge)
        return prompt

    def init_memory(self):
        """Initialize the memory for the Teaching Agent System."""
        # ai_prefix and human_prefix are swapped here compared to the TAS,
        # because the SSA is the student.
        memory = ConversationBufferMemory(memory_key="chat_history",
                                              return_messages=True,
                                              ai_prefix="Student",
                                              human_prefix="Teaching Agent System")
        return memory



    def build_ssa_v0(self):
        """Create the simulated student agent chain."""
        ssa_v0_memory = self.init_memory()
        ssa_v0_chain = LLMChain(llm=self.llm_model,
                                prompt=self.ssa_prompt,
                                memory=ssa_v0_memory,
                                #output_parser=BaseLLMOutputParser(),
                                verbose=False,)
        return ssa_v0_chain

    def predict(self, query):
        """Invoke the Teaching Agent System."""
        response = self.ssa_executor.invoke({"input": query})["text"]
        return response




if __name__ == '__main__':
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name = "Simulated Student Agent TEST 1")
    ssa = SSA(llm_model=llm_model)
    ssa_v0 = ssa.build_ssa_v0()
    response = ssa_v0.invoke({"input": "Hello I am August, your teacher!",})
    print(response)
    # print(ssa_v0.invoke({"input": "Can you try to explain how gradient descent works?",}))
    # print(ssa_v0.invoke({"input": "What was my name?",}))
    # print(ssa_v0.invoke({"input": "No, my name is not Felix, that's your name. What was my name?",}))







# # This is flexible, but you can define your agent here, or call your agent API here.
# def my_chat_bot(messages: List[dict]) -> dict:
#     system_message = {
#         "role": "system",
#         "content": """You are a STEM teaching assistant. \
#                     Your goal is firstly to explain subjects of interest. \
#                     Secondly, you should probe the students understanding of the subject by \
#                     asking questions that can help a bystanding teacher assess how well the student is doing. \
#                         The relevant subject is Deep Learning and the topic is gradient descent. """,
#     }
#     messages = [system_message] + messages
#     completion = openai.chat.completions.create(
#         messages=messages, model="gpt-3.5-turbo"
#     )
#     return completion.choices[0].message.model_dump()

# def chat_bot_node(messages):
#     # Convert from LangChain format to the OpenAI format, which our chatbot function expects.
#     messages = [convert_message_to_dict(m) for m in messages]
#     # Call the chat bot
#     chat_bot_response = my_chat_bot(messages)
#     # Respond with an AI Message
#     return AIMessage(content=chat_bot_response["content"])

# def _swap_roles(messages):
#     new_messages = []
#     for m in messages:
#         if isinstance(m, AIMessage):
#             new_messages.append(HumanMessage(content=m.content))
#         else:
#             new_messages.append(AIMessage(content=m.content))
#     return new_messages


# def should_continue(messages):
#     if len(messages) > 6:
#         return "end"
#     elif messages[-1].content == "FINISHED":
#         return "end"
#     else:
#         return "continue"



# if __name__ == "__main__":

#     llm_ver = init_llm_langsmith(llm_key=3)

#     simulated_student_agent_class = SimulatedStudentAgent(llm=llm_ver)
#     simulated_student_agent_chain = simulated_student_agent_class.create_simulated_student_agent_chain()

#     # output = simulated_student_agent_chain.invoke({"messages": [HumanMessage(content="Hi! How can I help you?")]})
#     # print(output)

#     def simulated_user_node(messages):
#         # Swap roles of messages
#         new_messages = _swap_roles(messages)
#         # Call the simulated user
#         response = simulated_student_agent_chain.invoke({"messages": new_messages})
#         # This response is an AI message - we need to flip this to be a human message
#         return HumanMessage(content=response.content)


#     my_chat_bot([{"role": "user", "content": "hi!"}])

#     graph_builder = MessageGraph()
#     graph_builder.add_node("user", simulated_user_node)
#     graph_builder.add_node("chat_bot", chat_bot_node)
#     # Every response from  your chat bot will automatically go to the
#     # simulated user
#     graph_builder.add_edge("chat_bot", "user")
#     graph_builder.add_conditional_edges(
#         "user",
#         should_continue,
#         # If the finish criteria are met, we will stop the simulation,
#         # otherwise, the virtual user's message will be sent to your chat bot
#         {
#             "end": END,
#             "continue": "chat_bot",
#         },
#     )
#     # The input will first go to your chat bot
#     graph_builder.set_entry_point("chat_bot")
#     simulation = graph_builder.compile()

#     for chunk in simulation.stream([]):
#         # Print out all events aside from the final end chunk
#         if END not in chunk:
#             print(chunk)
#             print("----")

