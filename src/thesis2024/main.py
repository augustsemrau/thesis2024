"""Main script for the thesis project.

Currently as of Wed 27/3, this script acts as a skeleton for the chatbot interactions."""

import sys
import pprint
import time
import os

## Local imports
from thesis2024.datamodules.crag import Crag



class SurrogateUser:
    def predict(self, question):
        # Implement the prediction logic for your LLM model here.
        # For example, sending a request to a model API or using a local model.
        return "What is the time now?"  # Dummy answer, replace with actual prediction logic.

class YourLLMModel:
    def __init__(self,
                llm: str = "gpt-3.5-turbo-0125",
                conversation_history: str = None,
                ):
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



class MainChat:
    """Main class for user-agent interactions.

    This class is responsible for letting the user and the agent interact with each other.
    Meanwhile, this class facilitates the recording and storing of the conversation history.
    """

    def __init__(self,
                user_system = SurrogateUser(),
                agent_system = YourLLMModel(),
                user_id: str = "User1",
                ):

        self.user = user_system
        self.user_ID = user_id
        self.conversation_history = self.get_conversation_history()
        # Initiate agent with the specific user
        self.agent_system = YourLLMModel(user_ID=self.user_ID)


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
    chat = MainChat(user_system=SurrogateUser(), agent_system=YourLLMModel(), user_ID="user_001")
    chat.run_chat()






























    # # Build the graph
    # crag_class = Crag()
    # app = crag_class.build_rag_graph()
    # print(app)
    # # Run the graph
    #     # Run
    # inputs = {"keys": {"question": "Who is the teacher of the machine learning course, and how come the highest mountains are located in asia?"}}
    # for output in app.stream(inputs):
    #     for key, value in output.items():
    #         # Node
    #         pprint.pprint(f"Node '{key}':")
    #         # Optional: print full state at each node
    #         # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    #     pprint.pprint("\n---\n")

    # # Final generation
    # pprint.pprint(value["keys"]["generation"])

    # print("Graph run complete.")
    # # sys.exit(0)

