from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl

import os
import getpass

from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv



def init_llm_langsmith(llm_key = 3):
    """Initialize the LLM model for the LangSmith system."""
    # Set environment variables
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass(f"Please provide your {var}")
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_API_KEY")
    # Add tracing in LangSmith.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if llm_key == 3:
        llm = "gpt-3.5-turbo-0125"
        os.environ["LANGCHAIN_PROJECT"] = "GPT-3.5 Conversation TEST"
    elif llm_key == 4:
        llm = "gpt-4-0125-preview"
        os.environ["LANGCHAIN_PROJECT"] = "GPT-4 Conversation TEST"
    return llm

llm_ver = init_llm_langsmith(llm_key=3)





# Load environment variables
load_dotenv('.env')

prompt_template = ChatPromptTemplate.from_messages(
    [
    ("system", 
    """
    You are a friendly assistant that answers user's question about astronomy.
    If the user's question is not about these topics, 
    respond with "Uh-oh! I do not have the information to answer your question. Ask me about Space, Planets and Stars!".
    """
    ),
    ("user", "{question}\n"),
    ]
)

@cl.on_chat_start
def main():
    # Instantiate required classes for the user session
    llm_chat = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_CHAT_MODEL"),
            temperature=0.0,
            max_tokens=100
    )
    llm_chain = LLMChain(prompt=prompt_template, llm=llm_chat, verbose=True)

    # Store the chain in the user session for reusability
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")
    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=res["text"]).send()


## chainlit run chainlit_conversational.py -w






# cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
# config = RunnableConfig(callbacks=[cb])
# result = await agent.ainvoke(input, config=config)

# model = ChatOpenAI(model_name=llm_ver, temperature=0, streaming=True)


# @cl.on_chat_start
# async def on_chat_start():
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
#             ),
#             ("human", "{question}"),
#         ]
#     )
#     runnable = prompt | model | StrOutputParser()
#     cl.user_session.set("runnable", runnable)


# @cl.on_message
# async def on_message(message: cl.Message):
#     runnable = cl.user_session.get("runnable")  # type: Runnable

#     msg = cl.Message(content="")

#     async for chunk in runnable.astream(
#         {"question": message.content},
#         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
#     ):
#         await msg.stream_token(chunk)

#     await msg.send()