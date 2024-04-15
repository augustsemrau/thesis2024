"""Frontend for conversing with the TAS, built using ChainLit."""
import chainlit as cl
from thesis2024.utils import init_llm_langsmith
from thesis2024.TAS import TAS

@cl.on_chat_start
def main():
    """Instantiate required classes for the user session."""
    llm_chat = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="CHAINLIT TEST 1")
    tas = TAS(llm_model=llm_chat)
    tas_v0 = tas.build_tas_v0()
    cl.user_session.set("tas", tas_v0)

@cl.on_message
async def main(message: str):
    """Call the TAS asynchronously and send the response to the user."""
    tas = cl.user_session.get("tas")
    res = await tas.ainvoke({"input": message.content,}, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["output"]).send()

