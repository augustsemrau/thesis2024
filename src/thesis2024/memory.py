"""Long-term memory for the system."""

# Basic Imports
import uuid
import csv
import os

# Local imports
from thesis2024.utils import init_llm_langsmith

# LangMem Imports
from langmem import AsyncClient, Client


def get_user_uuid_and_create_thread_id(user: str) -> str:
    """Get the UUID of the user if already existing, otherwise create and store new id."""
    user_uuid = None
    cwd_path = os.getcwd()
    file_path = cwd_path + "/data/langmem_data/user_uuid.csv"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == user:
                    user_uuid = row[1]
                    break
    if user_uuid is None:
        user_uuid = str(uuid.uuid4())
        with open(file_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow([user, user_uuid])

    user_name = f"{user}-{user_uuid[:4]}"

    thread_id = uuid.uuid4()
    return user_uuid, user_name, thread_id



class LongTermMemory:
    """Long-Term memory."""

    def __init__(self, user_name: str):
        """Initialize the long-term memory."""
        self.async_client = AsyncClient()
        self.client = Client()
        self.user_id, self.user_name, self.thread_id = get_user_uuid_and_create_thread_id(user=user_name)

    def get_user_data(self):
        """Retrieve data on the user."""
        return self.client.get_user(user_id=self.user_id)

    def save_conversation_step(self, user_query, llm_response):
        """Save a conversation step in the long-term memory."""
        conversation_step = [
        {
            "content": user_query,
            "role": "user",
            "name": self.user_name,
            "metadata": {"user_id": self.user_id,},
        },
        {
            "content": llm_response,
            "role": "assistant",
        }]
        self.client.add_messages(thread_id=self.thread_id, messages=conversation_step)
        self.client.trigger_all_for_thread(thread_id=self.thread_id)

    def get_user_memories(self, query: str):
        """Retrieve long term memories for the relevant user."""
        memories = self.client.query_user_memory(
                                                user_id=self.user_id,
                                                text=query,
                                                k=10,)
        # print(memories)
        # facts = "\n".join([mem["text"] for mem in memories["memories"]])
        # If no specific memory query is given, we sort by importance, else by relevance
        if query == "":
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["importance"], reverse=True)
        else:
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["relevance"], reverse=True)
        facts = ".\n".join([mem["text"] for mem in sorted_memories])
        return facts







if __name__ == "__main__":
    langsmith_name = "LangMem TEST 1 "
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name=langsmith_name)

