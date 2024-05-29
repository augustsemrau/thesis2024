"""Frontend for conversing with the TAS, built using ChainLit."""

# To run (from thesis2024/src/thesis2024/)
# chainlit run frontend/chainlit_conversational.py -w --port 7097
import time
import chainlit as cl
from thesis2024.utils import init_llm_langsmith
from thesis2024.TAS import TAS


"""Chainlit testing of the TAS."""
@cl.on_chat_start
def main():
    """Instantiate required classes for the user session."""
    tas_version = "v1"
    time_now = time.strftime("%Y.%m.%d-%H.%M.")
    langsmith_name = tas_version + " Chainlit-Conversation " + time_now
    llm_model = init_llm_langsmith(llm_key=40, temp=0.5, langsmith_name=langsmith_name)
    tas = TAS(llm_model=llm_model,
            version=tas_version,
            course="DeepLearning",#"IntroToMachineLearning",
            student_name="August",
            student_id="AugustSemrau001"
            )
    cl.user_session.set("tas", tas)

@cl.on_message
async def main(message: str):
    """Call the TAS asynchronously and send the response to the user."""
    tas = cl.user_session.get("tas")
    # response = tas.cl_predict(query=message.content)
    res = await tas.tas_executor.ainvoke({"input": message.content,}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    response = res["output"]
    await cl.Message(content=response).send()





"""Chainlit testing of the SSA."""
# @cl.on_chat_start
# def main():
#     """Instantiate required classes for the user session."""
#     llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="CHAINLIT SSA TEST 1")
#     ssa = SSA(llm_model=llm_model)
#     ssa_v0 = ssa.build_ssa_v0()
#     cl.user_session.set("ssa", ssa_v0)

# @cl.on_message
# async def main(message: str):
#     """Call the TAS asynchronously and send the response to the user."""
#     ssa = cl.user_session.get("ssa")
#     res = await ssa.ainvoke({"input": message.content,}, callbacks=[cl.AsyncLangchainCallbackHandler()])
#     await cl.Message(content=res["text"]).send()


