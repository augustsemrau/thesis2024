"""Long-term memory for the system."""


import openai
from langmem import AsyncClient, Client

import json
from typing import List
from pydantic import BaseModel, Field

# Local imports
from thesis2024.utils import init_llm_langsmith




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

    def __init__(self,
                 user_id: str):
        """Initialize the long-term memory."""
        self.oai_client = openai.AsyncClient()
        self.langmem_client = AsyncClient()

    def create_student_id(self):
        """Create new student ID and save it in a file for future reference."""

        pass

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

