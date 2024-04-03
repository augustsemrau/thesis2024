"""Main script for the thesis project.

Currently as of Wed 27/3, this script acts as a skeleton for the chatbot interactions."""


import time
import os
import getpass

## Local imports
from thesis2024.models.teaching_agent import TeachingAgent
from thesis2024.models.simulated_student_agent import SimulatedStudentAgent
from thesis2024.models.assessment_agent import AssessmentAgent


def init_llm_langsmith(llm_key = 3):
    """Initialize the LLM model for LangSmith.

    :param llm_key: Key for the LLM model to use.
    :return: LLM model name.
    """
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
        os.environ["LANGCHAIN_PROJECT"] = "GPT-3.5 Main System"
    elif llm_key == 4:
        llm = "gpt-4-0125-preview"
        os.environ["LANGCHAIN_PROJECT"] = "GPT-4 Main System"
    return llm


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




class MainSystem:
    """Main class for user-agent interactions.

    This class is responsible for letting the user and the agent interact with each other.
    Meanwhile, this class facilitates the recording and storing of the conversation history.
    """

    def __init__(self,
                student_system = TmpSimulatedStudentAgent(),
                teaching_system = TmpTeachingAgent(),
                assessment_system = TmpAssessmentAgent(),
                user_id: str = "User1",
                llm_ver = "gpt-3.5-turbo-0125"
                ):
        """Initialize the main system."""
        self.student_system = student_system
        self.agent_system = TmpTeachingAgent(user_ID=user_id)
        self.assessment_system = assessment_system
        self.conversation_history = self.get_conversation_history()


    def get_conversation_history(self):
        """Get the conversation history of the specific user."""
        # Compile all text files in the given directory
        user_conversation_history_path = f"data/conversations/{self.user_ID}/"
        self.conversation_history = []
        for file in os.listdir(user_conversation_history_path):
            with open(file, "r") as f:
                self.conversation_history.append(f.read())
        return self.conversation_history



    def run_chat(self):
        """Run the interaction between user and agent system."""
        conversation = []
        start_question = "Do you have a question (Y), or would you like the teaching agent to initiate the conversation (N)?"
        conversation.append("Start message: " + start_question + "\n")

        """Get the first answer"""
        # first_answer = self.user.predict(start_question)
        first_answer = "Y"
        conversation.append("User: " + first_answer + "\n")

        turn = 0
        """If the user has a question, the user initiates the conversation. Otherwise, the agent initiates the conversation."""
        if first_answer == "Y":
            response = self.user.predict("State your question.")
            turn = 0
            conversation.append("User: " + response + "\n")

        elif first_answer == "N":
            start_prompt = "Choose a sub-topic from the super-topic: 'Machine Learning'. Further, make sure to delve "
            " with topics present in previous conversation, if any, but only go into detail if the user clearly does "
            " not understand already delved with topics. "
            response = self.agent_system.predict(start_prompt)
            turn += 1
            conversation.append("Agent: " + response + "\n")


        """Main conversation loop."""
        while turn < 10 and response not in ["Q", "q"]:
            # Concat list of strings into one string
            current_conversation = " ".join(conversation[:-1])

            if turn % 2 != 0:
                response = self.user.predict(conversation[-1] + "Conversation history: " + current_conversation)
                conversation.append("User: " + response + "\n")
                turn += 1
            else:
                response = self.agent_system.predict(conversation[-1] + "Conversation history: " + current_conversation)
                conversation.append("Agent: " + response + "\n")
                turn += 1


        """Concatenate the conversation into one string, then update the conversation history for the specific user."""
        final_conversation = " ".join(conversation[:])
        print(final_conversation)








if __name__ == "__main__":

    llm = init_llm_langsmith(llm_key=3)

    # chat = MainSystem(student_system=SimulatedStudentAgent(),
    #                 teaching_system=TeachingAgent(),
    #                 assessment_system=AssessmentAgent(),
    #                 user_ID="user_001",
    #                 llm_ver=llm)
    main_system_class = MainSystem(student_system=TmpSimulatedStudentAgent(),
                    teaching_system=TmpTeachingAgent(),
                    assessment_system=TmpAssessmentAgent(),
                    user_ID="user_001",
                    llm_ver=llm)

    main_system_class.run_chat()


