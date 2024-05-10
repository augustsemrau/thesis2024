"""Long-term memory for the system."""

# Basic Imports
import uuid
import csv
import os
from typing import List

from pydantic import BaseModel, Field

# LangMem Imports
from langmem import AsyncClient, Client

# Local imports
from thesis2024.utils import init_llm_langsmith


"""A function that takes in a user name and creates a UUID for the user if the name is not already present in a csv file.
If it is, it returns the UUID associated with the user name.
If it is not, it generates a new UUID and saves it in the csv file."""
def get_user_uuid_and_create_thread_id(user: str) -> str:
    """Get the UUID of the user."""
    user_uuid = None
    file_path = "data/langmem_data/user_uuid.csv"
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



class Student(BaseModel):
    """A student in the system."""

    name: str = Field(default=None, description="The name of the student.")
    university: str = Field(
        default=None, description="The name of the university the student studies at.")
    current_education: str = Field(
        default=None, description="The name of the current education the student is undertaking.")
    current_year: int = Field(
        default=None, description="The current year of the student's education.")


class UserProfile(BaseModel):
    """A user profile in the system."""

    preferred_name: str = Field(default=None, description="The users's name.")
    summary: str = Field(
        default="",
        description="A quick summary of how the user would describe themselves.",)
    interests: List[str] = Field(
        default_factory=list,
        description="Short (two to three word) descriptions of areas of particular interest for the user. This can be a concept, activity, or idea. Favor broad interests over specific ones.",)
    education: List[Student] = Field(
        default_factory=Student,
        description="A summary of prior and present education.",)
    other_info: List[str] = Field(
        default_factory=list,
        description="",)

async def create_student_profile_memory():
    langmem_client = AsyncClient()
    student_profile_memory = await langmem_client.create_memory_function(
        UserProfile, target_type="user_state")





class CoreBelief(BaseModel):
    belief: str = Field(
        default="",
        description="The belief the user has about the world, themselves, or anything else.",)
    why: str = Field(description="Why the user believes this.")
    context: str = Field(
        description="The raw context from the conversation that leads you to conclude that the user believes this.")
async def create_student_belief_memory():
    langmem_client = AsyncClient()
    belief_function = await langmem_client.create_memory_function(
        CoreBelief, target_type="user_append_state")





class FormativeEvent(BaseModel):
    event: str = Field(
        default="",
        description="The event that occurred. Must be important enough to be formative for the user.",)
    impact: str = Field(default="", description="How this event influenced the user.")
# oai_client = openai.AsyncClient()
async def create_student_formative_event_memory():
    langmem_client = AsyncClient()

    event_function = await langmem_client.create_memory_function(
        FormativeEvent, target_type="user_append_state")



class LongTermMemory:
    """Long-term memory."""

    def __init__(self, student_name: str):
        """Initialize the long-term memory."""
        self.langmem_client = AsyncClient()
        self.user_uuid, self.user_name, self.thread_id = get_user_uuid_and_create_thread_id(user=student_name)

    async def save_conversation_step(self, student_query, TAS_response):

        conversation_step = [
        {
            "role": "user",
            # Names are optional but should be consistent with a given user id, if provided
            "name": self.username,
            "content": "Hey johnny have i ever told you about my older bro steve?",
            "metadata": {
                "user_id": str(self.user_id),
            },
        },
        {
            "content": "no, you didn't, but I think he was friends with my younger sister sueann",
            "role": "user",
            "name": johnny_username,
            "metadata": {
                "user_id": str(johnny_user_id),
            },
        }]
        self.langmem_client.add_messages(thread_id=self.thread_id, messages=conversation_step)


    def get_core_beliefs(self):
        """Get the core beliefs of the student."""
        pass

    def get_formative_events(self):
        """Get the formative events of the student."""
        pass

    def get_longterm_memory(self):
        """Get the long-term memory of the student."""
        pass








if __name__ == "__main__":
    langsmith_name = "LangMem TEST 1 "
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name=langsmith_name)

