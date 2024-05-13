"""Long-term memory for the system."""

# Basic Imports
import uuid
import csv
import os

# Local imports
from thesis2024.utils import init_llm_langsmith

# LangMem Imports
from langmem import AsyncClient, Client
from typing import List
from pydantic import BaseModel, Field

def get_user_uuid_and_create_thread_id(user: str) -> str:
    """Get the UUID of the user if already existing, otherwise create and store new id."""
    user_uuid = None
    past_thread_ids = []
    cwd_path = os.getcwd()
    # file_path = cwd_path + "/data/langmem_data/user_uuid.csv"
    file_path = cwd_path + "/data/langmem_data/user_thread_ids.csv"
    rows = []

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            rows = list(reader)
            for row in rows:
                if row[0] == user:
                    user_uuid = row[1]
                    past_thread_ids = row[2:]
                    break

    if user_uuid is None:
        user_uuid = str(uuid.uuid4())
        rows.append([user, user_uuid])
        # with open(file_path, "a") as file:
        #     writer = csv.writer(file)
        #     writer.writerow([user, user_uuid])
    user_name = f"{user}-{user_uuid[:4]}"

    # Write new thread id to the next available column of the user's row

    current_thread_id = str(uuid.uuid4())
    for row in rows:
        if row[0] == user:
            row.append(current_thread_id)
            break
    with open(file_path, "w") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    return user_uuid, user_name, current_thread_id, past_thread_ids

"""User State"""
class User(BaseModel):
    """A user in the system."""

    name: str = Field(default=None,
        description="The name of the user.",
        )
    education: str = Field(default=None,
        description="The education which the user undertakes.",
        )

class UserProfile(BaseModel):
    """A user's profile."""

    preferred_name: str = Field(default=None, 
        description="The user's name.",
        )
    summary: str = Field(default="",
        description="A quick summary of how the user would describe themselves.",
        )
    interests: List[str] = Field(default_factory=list,
        description="Short (two to three word) descriptions of areas of particular interest for the user. This can be a concept, activity, or idea. Favor broad interests over specific ones.",
        )
    # relationships: List[User] = Field(default_factory=user,
    #     description="A list of friends, family members, colleagues, and other relationships.",
    #     )
    other_info: List[str] = Field(default_factory=list,
        description="",
        )

"""User Append State"""
class CoreBelief(BaseModel):
    """A core belief of the user."""

    belief: str = Field(default="",
        description="The belief the user has about the world, themselves, or anything else.",
        )
    why: str = Field(default="",
        description="Why the user believes this.",
        )
    context: str = Field(default="",
        description="The raw context from the conversation that leads you to conclude that the user believes this."
        )
class FormativeEvent(BaseModel):
    """Formative events for the user."""

    event: str = Field(default="",
        description="The event that occurred. Must be important enough to be formative for the student.",
        )
    impact: str = Field(default="",
        description="How this event influenced the user."
        )

"""Thread State"""
class ConversationSummary(BaseModel):
    """Summary of a conversation."""

    title: str = Field(description="Distinct for the conversation.")
    summary: str = Field(description="High level summary of the interactions.")
    topic: List[str] = Field(description="Tags for topics discussed in this conversation.")




class LongTermMemory:
    """Long-Term Memory class."""

    def __init__(self, user_name: str):
        """Initialize the long-term memory."""
        self.async_client = AsyncClient()
        self.client = Client()
        self.user_id, self.user_name, self.thread_id, self.past_thread_ids = get_user_uuid_and_create_thread_id(user=user_name)
        self.user_semantic_memory_function = self.get_user_semantic_memory_function()
        # Create memory functions
        self.user_state_function = self.client.create_memory_function(UserProfile, target_type="user_state")
        self.belief_function = self.client.create_memory_function(CoreBelief, target_type="user_append_state")
        self.event_function = self.client.create_memory_function(FormativeEvent, target_type="user_append_state")
        self.thread_summary_function = self.client.create_memory_function(ConversationSummary, target_type="thread_summary")

    def get_user_semantic_memory_function(self):
        """Retrieve the user semantic memory function."""
        functions = self.client.list_memory_functions(target_type="user")
        for func in functions:
            if func["type"] == "user_semantic_memory":
                return func


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
        # TODO Comment this out, not necessary for actual use
        self.client.trigger_all_for_thread(thread_id=self.thread_id)

    def get_user_semantic_memories(self, query: str):
        """Retrieve long term memories for the relevant user."""
        memories = self.client.query_user_memory(
                        user_id=self.user_id,
                        text=query,
                        k=10,
                        memory_function_ids=[self.user_semantic_memory_function["id"]],
                        )
        print(memories)
        if query == "":
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["importance"], reverse=True)
        else:
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["relevance"], reverse=True)
        facts = ".\n".join([mem["text"] for mem in sorted_memories])
        return facts

    def get_user_state(self, query: str):
        """Retrieve long term memories for the relevant user."""
        # user_state = None
        # while not user_state:
        user_state = self.client.get_user_memory(
                    user_id=self.user_id,
                    memory_function_id=self.user_state_function["id"]
                    )
        print(user_state)
        return user_state

    def get_user_append_memories(self, query: str):
        """Retrieve long term memories for the relevant user."""
        memories = self.client.query_user_memory(
                        user_id=self.user_id,
                        text=query,
                        k=10,
                        memory_function_ids=[self.belief_function["id"], self.event_function["id"]],
                        )
        print(memories)
        if query == "":
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["importance"], reverse=True)
        else:
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["relevance"], reverse=True)
        facts = ".\n".join([mem["text"] for mem in sorted_memories])
        return facts


    # TODO Test thread summary retrieval
    def get_thread_summaries(self):
        """Retrieve the summaries for all threads."""
        thread_summaries = {}
        for thread_id in self.past_thread_ids:
            try:
                thread_summary = self.client.get_thread_memory(thread_id=thread_id, memory_function_id=self.thread_summary_function["id"])
                thread_summaries[thread_id] = thread_summary
            except Exception:
                print(f"No memories for thread id {thread_id}")
                continue
        print(thread_summaries)







if __name__ == "__main__":
    langsmith_name = "LangMem TEST 1 "
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name=langsmith_name)

