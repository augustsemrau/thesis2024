"""Utility functions for the thesis project."""

import os
import getpass
import time

from langchain_openai import ChatOpenAI

def init_llm_langsmith(llm_key = 3, temp = 0.5, langsmith_name: str = ""):
    """Initialize the LLM model and LangSmith tracing.

    Args:
    ----
    llm_key (int): The LLM model designation.
    temp (float): The temperature to use in the LLM model.
    langsmith_name (str): The name of the LangSmith project.

    Returns:
    -------
    ChatOpenAI: LLM model.

    """
    # Set environment variables
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass(f"Please provide your {var}")
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_API_KEY")
    os.environ["LANGMEM_API_URL"] = "https://long-term-memory-shared-for-6e67d43146acf1d71174b-vz4y4ooboq-uc.a.run.app"
    _set_if_undefined("LANGMEM_API_KEY")
    # _set_if_undefined("WOLFRAM_ALPHA_APPID")
    _set_if_undefined("TAVILY_API_KEY")

    if llm_key == 3:
        llm_ver = "gpt-3.5-turbo-0125"
    elif llm_key == 4:
        llm_ver = "gpt-4-turbo-2024-04-09"
    elif llm_key == 40:
        llm_ver = "gpt-4o-2024-05-13"

    if langsmith_name is not None:
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        time_now = time.strftime("%Y.%m.%d-%H.%M.")
        os.environ["LANGCHAIN_PROJECT"] = langsmith_name + "_LLM:" + llm_ver + "_Temp: " + str(temp) + "_Timestamp:" + time_now

    llm_model = ChatOpenAI(model_name=llm_ver, temperature=temp)
    return llm_model

