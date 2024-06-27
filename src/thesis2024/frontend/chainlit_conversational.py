"""Frontend for conversing with the TAS, built using ChainLit."""

# To run (from thesis2024/src/thesis2024/)
# chainlit run frontend/chainlit_conversational.py -w --port 7097
import chainlit as cl
from thesis2024.utils import init_llm_langsmith
from thesis2024.TAS import TAS
from thesis2024.multiagent_modules.reflexion_multiagent import ReflexionMultiAgent


# """Chainlit testing of the TAS."""
@cl.on_chat_start
def main():
    """Instantiate required classes for the user session."""
    student_name, student_course, student_subject = "August", "IntroToMachineLearning", "Linear Regression"
    student_learning_preferences = "I prefer formulas and math in order to understand technical concepts"
    # student_name, student_course, student_subject = "August", "IntroToMachineLearning", ""
    # student_learning_preferences = ""

    """TAS Setup"""
    llm_model = init_llm_langsmith(llm_key=4, temp=0.5, langsmith_name="TAS Chainlit-Conversation")
    tas = TAS(llm_model=llm_model,
            baseline_bool=False,
            course=student_course,
            subject=student_subject,
            learning_prefs=student_learning_preferences,
            student_name=student_name,
            student_id="AugustSemrau_PMAS_Test1",
            )
    cl.user_session.set("tas", tas)

@cl.on_message
async def main(message: str):
    """Call the TAS asynchronously and send the response to the user."""
    tas = cl.user_session.get("tas")
    query = message.content
    response = await tas.tas_executor.ainvoke({"input": query,}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    response = response["output"]
    tas.save_conversation_step(user_query=query, llm_response=response)
    await cl.Message(content=response).send()


"""Chainlit testing of Reflexion setup."""
# @cl.on_chat_start
# def main():
#     """Instantiate required classes for the user session."""
#     student_name, student_course, student_subject = "August", "IntroToMachineLearning", "Linear Regression"
#     student_learning_preferences = "I prefer formulas and math in order to understand technical concepts"

#     """Reflexion Setup"""
#     # reflexion_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="REFLEXION TEST")
#     llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="REFLEXION TEST")
#     # llm_model = init_llm_langsmith(llm_key=4, temp=0.5, langsmith_name="BASELINE_CHAIN")
#     reflexion = ReflexionMultiAgent(llm_model=llm_model,
#                                     reflexion_model=llm_model,
#                                     max_iter=1,
#                                     course=student_course,
#                                     )
#     cl.user_session.set("reflexion", reflexion)

# @cl.on_message
# async def main(message: str):
#     """Call the TAS asynchronously and send the response to the user."""
#     reflexion = cl.user_session.get("reflexion")
#     # response = tas.cl_predict(query=message.content)
#     res = await reflexion.reflexion_executor.ainvoke({"input": message.content,}, callbacks=[cl.AsyncLangchainCallbackHandler()])
#     response = res["output"]
#     await cl.Message(content=response).send()


