"""Utility functions for the thesis project."""

import os
import getpass

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

    if llm_key == 3:
        llm_ver = "gpt-3.5-turbo-0125"
    elif llm_key == 4:
        llm_ver = "gpt-4-turbo-2024-04-09"

    if langsmith_name is not None:
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = llm_ver + "_Temp: " + str(temp) + "_" + langsmith_name

    llm_model = ChatOpenAI(model_name=llm_ver, temperature=temp)
    return llm_model

