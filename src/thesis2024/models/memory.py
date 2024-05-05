import openai
from langmem import AsyncClient, Client


import json
from typing import List

from pydantic import BaseModel, Field



from thesis2024.utils import init_llm_langsmith



class Student(BaseModel):
    """A student in the system."""

    name: str = Field(default=None, description="The name of the student.")
    current_education: str = Field(
        default=None, description="The name of the current education the student is undertaking.")
    current_year: int = Field(
        default=None, description="The current year of the student's education.")
    subjects: str = Field(
        default=None, description="The subjects the student is studying.")


class StudentProfile(BaseModel):
    """A student profile in the system."""

    preferred_name: str = Field(default=None, description="The student's name.")

    summary: str = Field(
        default="",
        description="A quick summary of how the student would describe themselves.",
    )
    interests: List[str] = Field(
        default_factory=list,
        description="Short (two to three word) descriptions of areas of particular interest for the user. This can be a concept, activity, or idea. Favor broad interests over specific ones.",
    )
    education: List[Student] = Field(
        default_factory=Student,
        description="A list of friends, family members, colleagues, and other relationships.",
    )
    other_info: List[str] = Field(
        default_factory=list,
        description="",
    )

class CoreBelief(BaseModel):
    belief: str = Field(
        default="",
        description="The belief the user has about the world, themselves, or anything else.",
    )
    why: str = Field(description="Why the user believes this.")
    context: str = Field(
        description="The raw context from the conversation that leads you to conclude that the user believes this."
    )

class FormativeEvent(BaseModel):
    event: str = Field(
        default="",
        description="The event that occurred. Must be important enough to be formative for the user.",
    )
    impact: str = Field(default="", description="How this event influenced the user.")


# def create_student_memory():
#     oai_client = openai.AsyncClient()
#     langmem_client = AsyncClient()

#     user_profile_memory = await langmem_client.create_memory_function(
#         UserProfile, target_type="user_state")

#     belief_function = await langmem_client.create_memory_function(
#         CoreBelief, target_type="user_append_state")

#     event_function = await langmem_client.create_memory_function(
#         FormativeEvent, target_type="user_append_state")



class LongTermMemory:
    """Long-term memory."""

    def __init__(self):
        """Initialize the long-term memory."""
        self.oai_client = openai.AsyncClient()
        self.langmem_client = AsyncClient()



if __name__ == "__main__":
    langsmith_name = "LangMem TEST 1 "
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name=langsmith_name)

