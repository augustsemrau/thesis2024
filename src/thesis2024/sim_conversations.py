"""Script for generating conversations using the TAS and SSA."""

import time
import os

## Local imports
from thesis2024.utils import init_llm_langsmith
from thesis2024.TAS import TAS
from thesis2024.models.SSA import SSA



class TmpSimulatedStudentAgent:
    """Placeholder class for the simulated student agent system."""

    def __init__(self,
            llm: str = "gpt-3.5-turbo-0125",
            ):
        """Initialize the simulated student agent system."""
        self.llm = llm

    def predict(self, question):
        # Implement the prediction logic for your LLM model here.
        # For example, sending a request to a model API or using a local model.
        return "What is the time now?"  # Dummy answer, replace with actual prediction logic.

class TmpTeachingAgent:
    """Placeholder class for the teaching agent system."""

    def __init__(self,
                llm: str = "gpt-3.5-turbo-0125",
                conversation_history: str = None,
                ):
        """Initialize the teaching agent system."""
        self.llm = llm
        self.conversation_history = conversation_history
        self.initiate_agent_system()


    def initiate_agent_system(self):
        """Initiate the agent system."""
        # self.model = create_openai_functions_agent(llm=self.llm, conversation_history=self.conversation_history)
        pass

    def predict(self, question):
        """Predict the next response."""
        # Implement the prediction logic for your LLM model here.
        # For example, sending a request to a model API or using a local model.
        current_time = time.time()
        return str(current_time)  # Dummy answer, replace with actual prediction logic.

class TmpAssessmentAgent:
    """Placeholder class for the assessment agent system."""

    def __init__(self,
                llm: str = "gpt-3.5-turbo-0125",
                conversation_history: str = None,
                ):
        """Initialize the assessment agent system."""
        self.llm = llm
        self.conversation_history = conversation_history

    def create_assessment_chain(self):
        """Create the assessment chain."""
        # Implement the assessment chain creation logic here.
        pass




class ConversationSimulation:
    """Main class for user-agent interactions.

    This class is responsible for letting the user and the agent interact with each other.
    Meanwhile, this class facilitates the recording and storing of the conversation history.
    """

    def __init__(self,
                student_system,
                teaching_system,
                user_id: str = "User1",
                ):
        """Initialize the main system."""
        self.student_system = student_system
        self.teaching_system = teaching_system
        # self.conversation_history = self.get_conversation_history(user_id=user_id)


    # def get_conversation_history(self, user_id):
    #     """Get the conversation history of the specific user."""
    #     # Compile all text files in the given directory
    #     user_conversation_history_path = f"data/conversations/{user_id}/"
    #     self.conversation_history = []
    #     for file in os.listdir(user_conversation_history_path):
    #         with open(file, "r") as f:
    #             self.conversation_history.append(f.read())
    #     return self.conversation_history



    def simulate_conversation(self, student_initiate=True):
        """Simulate a conversation between the simulated student and TAS."""
        conversation = []


        """If the user has a question, the user initiates the conversation. Otherwise, the agent initiates the conversation."""
        if student_initiate:
            turn = 0
            start_query = "You can start your conversation with the teacher any way you want."
            conversation.append("Start message: " + start_query + "\n")
            response = self.student_system.predict(query=start_query)
            conversation.append("Simulated Student: " + response + "\n")
        else:
            turn = 1
            start_query = """Choose a sub-topic from the super-topic: 'Machine Learning'. Further, make sure to delve 
            with topics present in previous conversation, if any, but only go into detail if the student clearly does 
            not understand already delved with topics."""
            conversation.append("Start message: " + start_query + "\n")
            response = self.teaching_system.predict(query=start_query)
            conversation.append("TAS: " + response + "\n")


        """Main conversation loop."""
        while turn < 5:
            if turn % 2 != 0:
                response = self.student_system.predict(query=str(conversation[-1]))
                conversation.append("Simulated Student: " + response + "\n")
                turn += 1
            else:
                response = self.teaching_system.predict(query=str(conversation[-1]))
                conversation.append("TAS: " + response + "\n")
                turn += 1


        """Concatenate the conversation into one string, then update the conversation history for the specific user."""
        # final_conversation = " ".join(conversation[:])
        final_conversation = conversation
        print(final_conversation)








if __name__ == "__main__":

    TAS_llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name=None)
    TAS_class = TAS(llm_model=TAS_llm_model, version="v0")


    AAS_llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="Simulation Conversation 1")
    SSA_class = SSA(llm_model=AAS_llm_model)


    ConversationSimulation_class = ConversationSimulation(student_system=TAS_class,
                                            teaching_system=SSA_class,
                                            user_id="user_001")

    ConversationSimulation_class.simulate_conversation(student_initiate=False)










## Leftover code for having TAS-user conversation, but not finished
    # def simulate_conversation(self):
    #     """Run the interaction between user and agent system."""
    #     conversation = []
    #     start_question = "Do you have a question (Y), or would you like the teaching agent to initiate the conversation (N)?"
    #     conversation.append("Start message: " + start_question + "\n")

    #     """Get the first answer"""
    #     # first_answer = self.user.predict(start_question)
    #     first_answer = "Y"
    #     conversation.append("User: " + first_answer + "\n")

    #     turn = 0
    #     """If the user has a question, the user initiates the conversation. Otherwise, the agent initiates the conversation."""
    #     if first_answer == "Y":
    #         response = self.student_system.predict(query="State your question.")
    #         turn = 0
    #         conversation.append("User: " + response + "\n")

    #     elif first_answer == "N":
    #         start_query = """Choose a sub-topic from the super-topic: 'Machine Learning'. Further, make sure to delve 
    #         with topics present in previous conversation, if any, but only go into detail if the user clearly does 
    #         not understand already delved with topics."""
    #         response = self.teaching_system.predict(query=start_query)
    #         turn += 1
    #         conversation.append("Agent: " + response + "\n")


    #     """Main conversation loop."""
    #     while turn < 10 and response not in ["Q", "q"]:
    #         # Concat list of strings into one string
    #         current_conversation = " ".join(conversation[:-1])

    #         if turn % 2 != 0:
    #             response = self.student_system.predict(conversation[-1] + "Conversation history: " + current_conversation)
    #             conversation.append("User: " + response + "\n")
    #             turn += 1
    #         else:
    #             response = self.teaching_system.predict(conversation[-1] + "Conversation history: " + current_conversation)
    #             conversation.append("Agent: " + response + "\n")
    #             turn += 1


    #     """Concatenate the conversation into one string, then update the conversation history for the specific user."""
    #     final_conversation = " ".join(conversation[:])
    #     print(final_conversation)
