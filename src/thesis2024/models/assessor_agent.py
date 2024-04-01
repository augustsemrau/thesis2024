"""Functions for the agent model."""

# LangChain imports
from langchain import hub
from langchain_core.output_parsers import StrOutputParser


from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union

from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from typing_extensions import TypedDict

# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str



dl_02456_desc = "Machine perception of natural signals has improved a lot in the recent years thanks to "
"deep learning (DL). Improved image recognition with DL will make self-driving cars possible and is leading to more "
"accurate image-based medical diagnosis. Improved speech recognition and natural language processing with DL will "
"lead to many new intelligent applications within health-care and IT. Pattern recognition with DL in large datasets "
"will give new tools for drug discovery, condition monitoring and many other data-driven applications. "
"The purpose of this course is to give the student a detailed understanding of the deep artificial neural "
"network models, their training, computational frameworks for deployment on fast graphical processing units, "
"their limitations and how to formulate learning in a diverse range of settings. These settings include "
"classification, regression, sequences and other types of structured input and outputs and for reasoning in "
"complex environments. "


dl_02456_lo = "A student who has met the objectives of the course will be able to: "
" 1. Demonstrate knowledge of machine learning terminology such as likelihood function, maximum likelihood, "
" Bayesian inference, feed-forward, convolutional and Transformer neural networks, and error back propagation. "
" 2. Understand and explain the choices and limitations of a model for a given setting. "
" 3. Apply and analyze results from deep learning models in exercises and own project work. "
" 4. Plan, delimit and carry out an applied or methods-oriented project in collaboration with fellow students and "
" project supervisor. "
" 5. Assess and summarize the project results in relation to aims, methods and available data. "
" 6. Carry out the project and interpret results by use of computational framework for GPU programming such as "
" PyTorch. "
" 7. Structure and write a final short technical report including problem formulation, description of methods, "
" experiments, evaluation and conclusion. "
" 8. Organize and present project results at the final project presentation and in report. "
" 9. Read, evaluate and give feedback to work of other students. "


dl_02456_outline = "Course outline week 1-8: "
" 1. Introduction to statistical machine learning, feed-forward neural networks (FFNN) and error back-propagation. Part I do it yourself on pen and paper. "
" 2. Introduction to statistical machine learning, feed-forward neural networks (FFNN) and error back-propagation. Part II do it yourself in NumPy. "
" 3. Introduction to statistical machine learning, feed-forward neural networks (FFNN) and error back-propagation. Part III PyTorch. "
" 4. Convolutional neural networks (CNN) + presentation of student projects. "
" 5. Sequence modelling for text data with Transformers. "
" 6. Tricks of the trade and data science with PyTorch + Start of student projects. "
" 7. Variational learning and generative adversarial networks for unsupervised and semi-supervised learning. "
" 8. Reinforcement learning - policy gradient and deep Q-learning. "



class AssessorAgent:
    """Multi-Agent coding LangGraph model."""

    def __init__(self, llm: str="gpt-3.5-turbo-0125"):
        """Initialize the Assessor Agent class."""
        self.llm = llm
        return None

    # Helper functions
    def create_agent(self, tools, system_message):
        """Create arbitrary agent."""
        functions = [format_tool_to_openai_function(t) for t in tools]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        return prompt | self.llm.bind_functions(functions)




if __name__ == "__main__":

    coding_class = CodingMultiAgent(llm="gpt-3.5-turbo-0125")

    coding_graph = coding_class.instanciate_graph()
    print(coding_graph)










