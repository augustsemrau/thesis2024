"""Functions for the agent model."""

# LangChain imports
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

import os
import getpass



dl_02456_name = "Deep Learning"

dl_02456_desc = ("Course Description: Machine perception of natural signals has improved a lot in the recent years thanks to "
"deep learning (DL). Improved image recognition with DL will make self-driving cars possible and is leading to more "
"accurate image-based medical diagnosis. Improved speech recognition and natural language processing with DL will "
"lead to many new intelligent applications within health-care and IT. Pattern recognition with DL in large datasets "
"will give new tools for drug discovery, condition monitoring and many other data-driven applications. "
"The purpose of this course is to give the student a detailed understanding of the deep artificial neural "
"network models, their training, computational frameworks for deployment on fast graphical processing units, "
"their limitations and how to formulate learning in a diverse range of settings. These settings include "
"classification, regression, sequences and other types of structured input and outputs and for reasoning in "
"complex environments. ")

dl_02456_lo = ("Course Learning Objectives: A student who has met the objectives of the course will be able to: "
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
" 9. Read, evaluate and give feedback to work of other students. ")

dl_02456_outline = ("Course Outline week 1-8: "
" 1. Introduction to statistical machine learning, feed-forward neural networks (FFNN) and error back-propagation. Part I do it yourself on pen and paper. "
" 2. Introduction to statistical machine learning, feed-forward neural networks (FFNN) and error back-propagation. Part II do it yourself in NumPy. "
" 3. Introduction to statistical machine learning, feed-forward neural networks (FFNN) and error back-propagation. Part III PyTorch. "
" 4. Convolutional neural networks (CNN) + presentation of student projects. "
" 5. Sequence modelling for text data with Transformers. "
" 6. Tricks of the trade and data science with PyTorch + Start of student projects. "
" 7. Variational learning and generative adversarial networks for unsupervised and semi-supervised learning. "
" 8. Reinforcement learning - policy gradient and deep Q-learning. ")





class AssessorAgent:
    """Multi-Agent coding LangGraph model."""

    def __init__(self, llm: str="gpt-3.5-turbo-0125",
                 course_name: str = "Deep Learning",
                 course_description: str = "",
                 course_learning_objectives: str = "",
                 course_outline: str = ""):
        """Initialize the Assessor Agent class."""
        self.model = ChatOpenAI(model_name=llm, temperature=0.5)
        self.partial_prompt = self.partial_assessment_prompt(course_name=course_name,
                                                                        course_description=course_description,
                                                                        course_learning_objectives=course_learning_objectives,
                                                                        course_outline=course_outline)
        return None

    def partial_assessment_prompt(self, course_name: str = "Deep Learning",
                                course_description: str = "",
                                course_learning_objectives: str = "",
                                course_outline: str = ""):

        prompt = PromptTemplate(template=(
                    "You are an expert in assessing student understanding and capabilities of STEM courses. "
                    "The learning objectives of the course {course_name} are the following: "
                    "\n{course_learning_objectives}. "
                    "\n \nThe course description is as follows: \n{course_description} "
                    "\n \nThe course outline is as follows: \n{course_outline} "
                    "\n \nYou will assess the student's understanding of the course based on a conversation between "
                    "the student and a teaching assistant. Based on this conversation, you will evaluate how well "
                    "the student has understood the course material in relation to the learning objectives. "
                    "Keep in mind that the conversation may only relate to one or few of the learning objectives, "
                    "and that your assessment should focus on only the relevant learning objectives. "
                    "You will provide your assessment to the teacher, and give feedback on which subjects "
                    "should be revisited by the student in order to meet the relevant learning objectives. "
                    "Your feedback should be short and critical."
                    "Below is the conversation between the student and the teaching assistant: \n \n{conversation} \n \n"
                    ),
                    input_variables=["conversation"],
                    partial_variables={"course_name": course_name,
                                     "course_description": course_description,
                                     "course_learning_objectives": course_learning_objectives,
                                     "course_outline": course_outline}
        )
        # prompt.pretty_print()
        return prompt


    def create_assessment_chain(self):
        """Create assessment chain."""
        chain = self.partial_prompt | self.model | StrOutputParser()
        return chain






if __name__ == "__main__":

        # Set environment variables
    def _set_if_undefined(var: str):
        if not os.environ.get(var):
            os.environ[var] = getpass(f"Please provide your {var}")
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_API_KEY")
    # Add tracing in LangSmith.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Assessor Agent GPT-3.5 Turbo 0125"


    # llm = "gpt-3.5-turbo-0125"
    llm = "gpt-4-0125-preview"
    assessor_agent_class = AssessorAgent(llm=llm,
                                        course_name=dl_02456_name,
                                        course_description=dl_02456_desc,
                                        course_learning_objectives=dl_02456_lo,
                                        course_outline=dl_02456_outline)

    short_conversation = ("User: Hello "
    " Agent: Hello, how are you? "
    " User: Good, but can you help me understand what gradient descent is? "
    " Agent: Sure, I can help you with that. Gradient descent is an optimization algorithm used to minimize the cost function in machine learning. It works by iteratively updating the parameters of a model in the direction of the negative gradient of the cost function. This process is repeated until the algorithm converges to a local minimum of the cost function. Does that help? "
    " User: Yes, thank you for explaining that. Can you also explain how gradient descent is used in training neural networks? "
    " Agent: Of course! In training neural networks, gradient descent is used to update the weights and biases of the network in order to minimize the loss function. The gradient of the loss function with respect to the weights and biases is calculated using backpropagation, and then the weights and biases are updated in the direction of the negative gradient. This process is repeated for each batch of training data until the network converges to a set of weights and biases that minimize the loss function. Does that make sense? "
    " User: Yes, that makes sense. Thank you for explaining it so clearly. "
    " Agent: You're welcome! If you have any more questions, feel free to ask. I'm here to help. "
    " User: Thank you, I will keep that in mind.")

    assessor_agent_chain = assessor_agent_class.create_assessment_chain()
    assessment = assessor_agent_chain.invoke({"conversation": short_conversation})
    print("\n")
    print(assessment)










