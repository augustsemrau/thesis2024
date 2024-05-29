"""Module containing the Assessment Agent System (AAS)."""


# LangChain imports
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Local Imports
from thesis2024.utils import init_llm_langsmith


dl_02456_name = "Deep Learning"

dl_02456_lo = ("Course Learning Objectives: A student who has met the objectives of the course will be able to:"
"\n1. Demonstrate knowledge of machine learning terminology such as likelihood function, maximum likelihood, "
"Bayesian inference, feed-forward, convolutional and Transformer neural networks, and error back propagation."
"\n2. Understand and explain the choices and limitations of a model for a given setting."
"\n3. Apply and analyze results from deep learning models in exercises and own project work."
"\n4. Plan, delimit and carry out an applied or methods-oriented project in collaboration with fellow students and "
"project supervisor. "
"\n5. Assess and summarize the project results in relation to aims, methods and available data. "
"\n6. Carry out the project and interpret results by use of computational framework for GPU programming such as PyTorch."
"\n7. Structure and write a final short technical report including problem formulation, description of methods, "
"experiments, evaluation and conclusion."
"\n8. Organize and present project results at the final project presentation and in report."
"\n9. Read, evaluate and give feedback to work of other students.")

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

dl_02456_outline = ("Course Outline week 1-8: "
" 1. Introduction to statistical machine learning, feed-forward neural networks (FFNN) and error back-propagation. Part I do it yourself on pen and paper. "
" 2. Introduction to statistical machine learning, feed-forward neural networks (FFNN) and error back-propagation. Part II do it yourself in NumPy. "
" 3. Introduction to statistical machine learning, feed-forward neural networks (FFNN) and error back-propagation. Part III PyTorch. "
" 4. Convolutional neural networks (CNN) + presentation of student projects. "
" 5. Sequence modelling for text data with Transformers. "
" 6. Tricks of the trade and data science with PyTorch + Start of student projects. "
" 7. Variational learning and generative adversarial networks for unsupervised and semi-supervised learning. "
" 8. Reinforcement learning - policy gradient and deep Q-learning. ")





class AAS:
    """Class for Assessment Agent System."""

    def __init__(self, llm_model,
                 course_name: str = "Deep Learning",
                 course_description: str = "",
                 course_learning_objectives: str = "",
                 course_outline: str = ""):
        """Initialize the Assessor Agent class."""
        self.llm_model = llm_model
        self.course_name=course_name
        self.course_description=course_description
        self.course_learning_objectives=course_learning_objectives
        self.course_outline=course_outline
        return None

    def create_assessment_chain(self):
        """Create assessment chain."""
        assessment_prompt = PromptTemplate(template=(
                    "ROLE: You are a professor in the STEM course '{course_name}', and your goal is to grade students."
                    "\n \nThe learning objectives objectives of the course are the following:\n{course_learning_objectives}"
                    #"\n \nThe course description is as follows:\n{course_description}"
                    #"\n \nThe course outline is as follows:\n{course_outline}"
                    "\n \nINSTRUCTIONS: You will assess the student's understanding of the course based on a conversation between "
                    "the student and a teacher. Based on this conversation, you will evaluate how well "
                    "the student has understood the relevant course material in relation to the learning objectives. "
                    "Keep in mind that the conversation may only relate to one or few of the learning objectives, "
                    "and that your assessment should focus on only the relevant subjects."
                    "Further, if ther student does not explicitly show understanding of a subject, do not assume they do."
                    "You will provide your assessment to the teacher, not the student."#, and give feedback on which subjects "
                    #"should be revisited by the student in order to meet the relevant learning objectives. "
                    "Your feedback should be short and critical."
                    "\n \nBelow is the conversation between the student and the teaching assistant:\n{conversation}"
                    ),
                    input_variables=["conversation"],
                    partial_variables={"course_name": self.course_name,
                                     "course_description": self.course_description,
                                     "course_learning_objectives": self.course_learning_objectives,
                                     "course_outline": self.course_outline}
        )


        assessment_chain = assessment_prompt | self.model | StrOutputParser()


        """Create summary chain."""
        summary_prompt = PromptTemplate(template=(
                    "ROLE: You are an expert in summarizing teaching feedback."
                    "\n \nINSTRUCTIONS: You will summarize the feedback provided by the professor to the student."
                    "The feedback should be concise and focus on the key points of the assessment."
                    "You will provide a summary of the feedback to the student, no longer than two sentences."
                    "\n \nBelow is the feedback provided by the professor:\n{feedback}"
                    ),
                    input_variables=["feedback"]
        )

        summary_chain = ({"feedback": assessment_chain} | summary_prompt | self.llm_model | StrOutputParser())

        return summary_chain






if __name__ == "__main__":


    llm_model = llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="AAS TEST 1")


    AAS_class = AAS(llm_model=llm_model,
                                    course_name=dl_02456_name,
                                    course_description=dl_02456_desc,
                                    course_learning_objectives=dl_02456_lo,
                                    course_outline=dl_02456_outline)

    short_conversation = ("Student: Hello \n"
    " Teacher: Hello, how are you? \n"
    " Student: Good, but can you help me understand what gradient descent is? \n"
    " Teacher: Sure, I can help you with that. Gradient descent is an optimization algorithm used to minimize the cost function in machine learning. It works by iteratively updating the parameters of a model in the direction of the negative gradient of the cost function. This process is repeated until the algorithm converges to a local minimum of the cost function. Does that help? \n"
    " Student: Yes, thank you for explaining that. Can you also explain how gradient descent is used in training neural networks? \n"
    " Teacher: Of course! In training neural networks, gradient descent is used to update the weights and biases of the network in order to minimize the loss function. The gradient of the loss function with respect to the weights and biases is calculated using backpropagation, and then the weights and biases are updated in the direction of the negative gradient. This process is repeated for each batch of training data until the network converges to a set of weights and biases that minimize the loss function. Does that make sense? \n"
    " Student: Yes, that makes sense. Thank you for explaining it so clearly. \n"
    " Teacher: You're welcome! If you have any more questions, feel free to ask. I'm here to help. \n"
    " Student: Thank you, I will keep that in mind.")

    # short_conversation = """User: I am struggeling with the following problem: I have a list of 1000 numbers and I want to find the 10 largest numbers in the list. How can I do this?
    # Agent: You can use the following code to find the 10 largest numbers in a list of 1000 numbers: 
    # ```python
    # import heapq

    # def find_largest_numbers(numbers, k):
    #     return heapq.nlargest(k, numbers)

    # numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # k = 10
    # largest_numbers = find_largest_numbers(numbers, k)
    # print(largest_numbers)
    # ```
    # User: Thank you! This code works perfectly.
    # Agent: You're welcome! If you have any more questions or need further assistance, feel free to ask."""


    assessor_agent_chain = AAS_class.create_assessment_chain()
    summarized_assessment = assessor_agent_chain.invoke({"conversation": short_conversation})
    print("\n")
    print(summarized_assessment)
    print("\n")
    print("Assessment completed.")















