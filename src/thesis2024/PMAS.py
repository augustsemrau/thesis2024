"""Long-term memory for the system."""

# Basic Imports
import uuid
import csv
import os

# LangChain Imports
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

# LangMem Imports
from langmem import AsyncClient, Client
from typing import List
from pydantic import BaseModel, Field

# Local imports
from thesis2024.utils import init_llm_langsmith
from thesis2024.tools import ToolClass


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
class UserProfile(BaseModel):
    """A user's profile."""

    student_name: str = Field(default=None,
        description="The name of the student.",
        )
    summary: str = Field(default="",
        description="A quick summary of how the user would describe themselves.",
        )
    education: str = Field(default=None,
        description="The education which the studernt undertakes.",
        )
    courses: List[str] = Field(default_factory=list,
        description="The courses the student is currently taking.",
        )
    learning_preferences: str = Field(default="",
        description="The learning preferences of the student."
        )

"""User Append State"""
class LearningPreference(BaseModel):
    """A learning preference of the user."""

    learning_preference: str = Field(default="",
        description="The learning preference.",
        )
    why: str = Field(default="",
        description="Why the user prefers this way of teaching.",
        )
    context: str = Field(default="",
        description="The context and perhaps courses/subjects in which this learning prefernce is apparent.",
        )
    source_comment: str = Field(default="",
        description="The raw user utterance where you identified this preference.",
        )

class SubjectComprehension(BaseModel):
    """Formative events for the user."""

    subject: str = Field(default="",
        description="The subject which the student discussed in this interaction/conversation.",
        )
    comprehension: str = Field(default="",
        description="How well did the user comprehend this subject? Did they show clear comprehension, a lack of comprehension, or not show whether their comprehend the subject?"
        )

"""Thread State"""
class ConversationSummary(BaseModel):
    """Summary of a conversation."""

    title: str = Field(default="",
        description="Concise 2-5 word title for conversation.",
        )
    summary: str = Field(default="",
        description="High level summary of the interactions.",
        )
    topic: List[str] = Field(default_factory=list,
        description="Tags for topics discussed in this conversation.",
        )



class LongTermMemory:
    """Long-Term Memory class."""

    def __init__(self, user_name: str):
        """Initialize the long-term memory."""
        self.async_client = AsyncClient()
        self.client = Client()
        self.user_id, self.user_name, self.thread_id, self.past_thread_ids = get_user_uuid_and_create_thread_id(user=user_name)

        self.user_semantic_memory_function = self.get_user_semantic_memory_function()

        # Create memory functions
        ## User State
        self.user_state_function = self.client.create_memory_function(UserProfile,
                                                                      target_type="user_state")
        ## User Append State
        self.user_learning_preference_function = self.client.create_memory_function(LearningPreference,
                                                                                    target_type="user_append_state",
                                                                                    custom_instructions="Identify what learning preferences the student may have, which are apparent in the conversation.")
        self.subject_comprehension_function = self.client.create_memory_function(SubjectComprehension,
                                                                                 target_type="user_append_state",
                                                                                 custom_instructions="Extract the subjects delved with in the conversation.")
        ## Thread State
        self.thread_summary_function = self.client.create_memory_function(ConversationSummary,
                                                                          target_type="thread_summary")

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
        self.client.add_messages(thread_id=self.thread_id,
                                 messages=conversation_step)
        # TODO Comment this out, not necessary for actual use
        self.client.trigger_all_for_thread(thread_id=self.thread_id)

    def get_user_semantic_memories(self, query: str):
        """Retrieve long term semantic memories for the relevant user."""
        memories = self.client.query_user_memory(
                        user_id=self.user_id,
                        text=query,
                        k=10,
                        memory_function_ids=[self.user_semantic_memory_function["id"]],
                        )
        # print(memories)
        if query == "":
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["importance"], reverse=True)
        else:
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["relevance"], reverse=True)
        facts = ".\n".join([mem["text"] for mem in sorted_memories])
        return facts

    def get_user_state(self):
        """Retrieve long term memories for the relevant user."""
        # user_state = None
        # while not user_state:
        user_state = self.client.get_user_memory(
                    user_id=self.user_id,
                    memory_function_id=self.user_state_function["id"]
                    )
        print("\n\nUser Profile: ", user_state)
        return user_state

    def get_learning_preference_memories(self, query: str = ""):
        """Retrieve long term memories for the relevant user."""
        memories = self.client.query_user_memory(
                        user_id=self.user_id,
                        text=query,
                        k=10,
                        memory_function_ids=[self.user_learning_preference_function["id"]],
                        )
        print("\n\nLearning preference memories: ",memories)
        if query == "":
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["importance"], reverse=True)
        else:
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["relevance"], reverse=True)
        facts = ".\n".join([mem["text"] for mem in sorted_memories])
        return facts

    def get_subject_comprehension_memories(self, query: str = ""):
        """Retrieve long term memories for the relevant user."""
        memories = self.client.query_user_memory(
                        user_id=self.user_id,
                        text=query,
                        k=10,
                        memory_function_ids=[self.subject_comprehension_function["id"]],
                        )
        print("\n\nSubject comprehension memories: ", memories)
        if query == "":
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["importance"], reverse=True)
        else:
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["relevance"], reverse=True)
        facts = ".\n".join([mem["text"] for mem in sorted_memories])
        return facts

    def get_subject_comprehension(self, query: str = ""):
        """Retrieve long term memories for the relevant user."""
        memories = self.client.get_user_memory(
                        user_id=self.user_id,
                        # text=query,
                        # k=10,
                        memory_function_ids=[self.subject_comprehension_function["id"]],
                        )
        print("\n\nSubject comprehension memories: ", memories)
        if query == "":
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["importance"], reverse=True)
        else:
            sorted_memories = sorted(memories["memories"], key=lambda x: x["scores"]["relevance"], reverse=True)
        facts = ".\n".join([mem["text"] for mem in sorted_memories])
        return facts

    def get_thread_summaries(self):
        """Retrieve the summaries for all threads."""
        thread_summaries = {}
        for thread_id in self.past_thread_ids:
            try:
                thread_summary = self.client.get_thread_memory(thread_id=thread_id,
                                                               memory_function_id=self.thread_summary_function["id"])
                thread_summaries[thread_id] = thread_summary
                print(f"Thread id {thread_id} has the following summary: {thread_summary}")

            except Exception:
                # print(f"No memories for thread id {thread_id}")
                continue
        return thread_summaries

    def predict(self, query: str):
        """Invoke the Long-Term Memory to retrieve all TAS-relevant personal memory."""
        tas_relevant_memories = ""
        tas_relevant_memories += f"Student Profile: {self.get_user_state()}\n\n"
        tas_relevant_memories += f"Learning Preferences for the Student: {self.get_learning_preference_memories(query=query)}\n\n"
        tas_relevant_memories += f"These are the Subjects the Student has Interacted with: {self.get_subject_comprehension_memories(query=query)}\n\n"

        if tas_relevant_memories == "":
            return "No relevant memories found in all local data."
        else:
            return tas_relevant_memories





class PMAS:
    """Personal Memory Agent System (PMAS)."""

    def __init__(self, llm_model,
                 student_name,
                 course,
                 subject):
        """Initialize the PMAS."""
        self.llm_model = llm_model
        self.course = course
        self.pmas_prompt = self.build_pmas_prompt(student_name=student_name,
                                                course_name=course,
                                                subject_name=subject)
        self.pmas_executor = self.build_pmas()


    def build_pmas_prompt(self, student_name, course_name, subject_name):
        """Build the agent prompt."""
        prompt_hub_template = hub.pull("augustsemrau/pmas-agent-prompt").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial(student_name=student_name,
                                         course_name=course_name,
                                         subject_name=subject_name,
                                        )
        return prompt

    def build_pmas(self):
        """Build the Teaching Agent System (TAS)."""
        tool_class = ToolClass()
        tools = [tool_class.build_search_tool(),
                 tool_class.build_retrieval_tool(course_name=self.course),
                ]

        pmas_agent = create_react_agent(llm=self.llm_model,
                                       tools=tools,
                                       prompt=self.pmas_prompt,
                                       output_parser=None)
        pmas_agent_executor = AgentExecutor(agent=pmas_agent,
                                           tools=tools,
                                        #    memory=self.short_term_memory,
                                           verbose=True,
                                           handle_parsing_errors=True)
        return pmas_agent_executor

    def predict(self, conversation):
        """Invoke the PMAS."""
        # print("\n\nInteraction:\n", interaction)
        personal_memory = self.pmas_executor.invoke({"input": conversation})["output"]
        # print("\n\nPersonal Memory:\n", personal_memory)
        return personal_memory




if __name__ == "__main__":
    langsmith_name = "LangMem TEST 1 "
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name=langsmith_name)
    ltm_class = LongTermMemory(user_name="AugustSemrau2")
    ltm_class.past_thread_ids




# class User(BaseModel):
#     """A user in the system."""

#     name: str = Field(default=None,
#         description="The name of the user.",
#         )
#     education: str = Field(default=None,
#         description="The education which the user undertakes.",
#         )
#     courses: List[str] = Field(default_factory=list,
#         description="The courses the user is currently taking.",
#         )


# self.belief_function = self.client.create_memory_function(CoreBelief, target_type="user_append_state")
# self.event_function = self.client.create_memory_function(FormativeEvent, target_type="user_append_state")

# class CoreBelief(BaseModel):
#     """A core belief of the user."""

#     belief: str = Field(default="",
#         description="The belief the user has about the world, themselves, or anything else.",
#         )
#     why: str = Field(default="",
#         description="Why the user believes this.",
#         )
#     context: str = Field(default="",
#         description="The raw context from the conversation that leads you to conclude that the user believes this."
#         )
# class FormativeEvent(BaseModel):
#     """Formative events for the user."""

#     event: str = Field(default="",
#         description="The event that occurred. Must be important enough to be formative for the student.",
#         )
#     impact: str = Field(default="",
#         description="How this event influenced the user."
#         )


# from typing import List
# from pydantic import BaseModel, Field
# # LangMem Imports
# from langmem import AsyncClient, Client


# def get_core_beliefs(self):
#     """Get the core beliefs of the student."""
#     pass

# def get_formative_events(self):
#     """Get the formative events of the student."""
#     pass

# def get_longterm_memory(self):
#     """Get the long-term memory of the student."""
#     pass

# class Student(BaseModel):
#     """A student in the system."""

#     name: str = Field(default=None, description="The name of the student.")
#     university: str = Field(
#         default=None, description="The name of the university the student studies at.")
#     current_education: str = Field(
#         default=None, description="The name of the current education the student is undertaking.")
#     current_year: int = Field(
#         default=None, description="The current year of the student's education.")

# class UserProfile(BaseModel):
#     """A user profile in the system."""

#     preferred_name: str = Field(default=None, description="The users's name.")
#     summary: str = Field(
#         default="",
#         description="A quick summary of how the user would describe themselves.",)
#     interests: List[str] = Field(
#         default_factory=list,
#         description="Short (two to three word) descriptions of areas of particular interest for the user. This can be a concept, activity, or idea. Favor broad interests over specific ones.",)
#     education: List[Student] = Field(
#         default_factory=Student,
#         description="A summary of prior and present education.",)
#     other_info: List[str] = Field(
#         default_factory=list,
#         description="",)

# async def create_student_profile_memory():
#     langmem_client = AsyncClient()
#     student_profile_memory = await langmem_client.create_memory_function(
#         UserProfile, target_type="user_state")

# class CoreBelief(BaseModel):
#     belief: str = Field(
#         default="",
#         description="The belief the user has about the world, themselves, or anything else.",)
#     why: str = Field(description="Why the user believes this.")
#     context: str = Field(
#         description="The raw context from the conversation that leads you to conclude that the user believes this.")

# async def create_student_belief_memory():
#     langmem_client = AsyncClient()
#     belief_function = await langmem_client.create_memory_function(
#         CoreBelief, target_type="user_append_state")

# class FormativeEvent(BaseModel):
#     event: str = Field(
#         default="",
#         description="The event that occurred. Must be important enough to be formative for the user.",)
#     impact: str = Field(default="", description="How this event influenced the user.")
# # oai_client = openai.AsyncClient()
# async def create_student_formative_event_memory():
#     langmem_client = AsyncClient()

#     event_function = await langmem_client.create_memory_function(
#         FormativeEvent, target_type="user_append_state")
