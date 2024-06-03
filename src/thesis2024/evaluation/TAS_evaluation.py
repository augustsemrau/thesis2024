"""Module contains the implementation of the TAS evaluation."""

import time


# Langchain imports
from langchain import chat_models, prompts, smith, hub
from langchain.schema import output_parser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Langsmith imports
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

# Local imports
from thesis2024.utils import init_llm_langsmith



class OtherEvaluationMetrics:
    """Class for other evaluation metrics."""

    def __init__(self):
        """Initialize."""
        pass

    def is_answered(self, run: Run, example: Example) -> dict:
        """Check if the question is answered."""
        conversation = run.outputs.get("Conversation")
        if not conversation:
            return {"key": "is_answered" , "score": 0}
        else:
            return {"key": "is_answered" , "score": 1}

    def conversation_length(self, run: Run, example: Example) -> dict:
        """Check the length of the conversation."""
        conversation = run.outputs.get("Conversation")
        return {"key": "conversation_length" , "score": len(conversation)}


class TasEvaluator:
    """Class for evaluating the TAS."""

    def __init__(self, eval_model, experiment_name: str, few_shot_examples: str):
        """Initialize."""
        self.eval_model = eval_model
        self.experiment_name = experiment_name
        self.few_shot_examples = few_shot_examples


    def build_eval_prompt(self, prompt_name: str, criteria: str, criteria_des: str):
        """Prompt for the TAS Evaluator."""
        prompt_hub_template = hub.pull(prompt_name).template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial(criteria_name=criteria,
                                         criteria_description=criteria_des,
                                         few_shot_examples=self.few_shot_examples)
        return prompt


    def output_dataset(self, inputs: dict) -> dict:
        """Parse the dataset to extract the chat history which is evaluated upon."""
        chat_hist = inputs["chat_history"]
        # second_last_message = inputs["input"]
        # outputs = run.outputs["output"]

        conversation = str(chat_hist) #+ "\n" + second_last_message

        self.personalization_output = self.personalization(conversation=conversation)
        # self.engagement_output = self.engagement(conversation=conversation)
        # self.repetition_output = self.repetition(conversation=conversation)
        # self.correctness_output = self.correctness(conversation=conversation)
        # self.relevance_output = self.relevance(conversation=conversation)
        # self.clarity_output = self.clarity(conversation=conversation)
        # self.adaptability_output = self.adaptability(conversation=conversation)

        return {
                "Personalization": self.personalization_output,
                # "Engagement": self.engagement_output,
                # "Repetition": self.repetition_output,
                # "Correctness": self.correctness_output,
                # "Relevance": self.relevance_output,
                # "Clarity": self.clarity_output,
                # "Adaptability": self.adaptability_output,
                "Conversation": conversation}


    def run_evaluation(self, dataset_name):
        """Run the evaluation experiment."""
        other_metrics = OtherEvaluationMetrics()
        evalulators = [other_metrics.is_answered,
                       other_metrics.conversation_length,
                       self.personalization_grade,
                    #    self.engagement_grade,
                    #    self.repetition_grade,
                    #    self.correctness_grade,
                    #    self.relevance_grade,
                    #    self.clarity_grade,
                    #    self.adaptability_grade,
                       ]
        # Run
        experiment_results = evaluate(self.output_dataset,
                data=dataset_name,
                evaluators=evalulators,
                experiment_prefix=self.experiment_name,
                # Any experiment metadata can be specified here
                # metadata={"variant": "stuff website context into gpt-3.5-turbo",},
                )
        return experiment_results


    def personalization(self, conversation):
        """Evaluate correctness."""
        criteria = "Personalization"
        criteria_des = """
The student states their personal learning preferences, and the Teaching Assistant should adapt to these preferences. Adapting it's explanations to these learning preferences is paramount. 
How well does the Teaching Assistant personalize it's explanation of the given subject to the student? 
Better personalization be given a higher score, and neglegting the student's learning preferences should be given a lower score.
"""
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def personalization_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Personalization", "score": float(self.personalization_output['Grade'])}



    def engagement(self, conversation):
        """Evaluate correctness."""
        criteria = "Engagement"
        criteria_des = """
The teaching assistant should engage the student in the conversation, making it interesting and interactive. Asking questions, providing examples, and encouraging participation are all ways to increase engagement.
How engaging is the Teaching Assistant in the conversation?
More engaging should be given a higher score, and less engaging should be given a lower score.
"""
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def engagement_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Engagement", "score": float(self.engagement_output['Grade'])}



    def repetition(self, conversation):
        """Evaluate correctness."""
        criteria = "Repetition"
        criteria_des = """
Repeating the same information multiple times can be necessary in order for the student to understand it, but should be conveyed in alternative ways by rephrasing and recontextualing.
How well is the Teaching Assistant conveying the information without repeating itself?
Avoiding the repeating of information should be given a higher score. If repeats are done, the ability to rephrase the same information in new ways should be given a higher score. Repeating information without altering it should be given a lower score.
"""
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def repetition_grade(self, run: Run, example: Example) -> dict:
        """Output repetition grade to Langsmith."""
        return {"key": "Repetition", "score": float(self.repetition_output['Grade'])}



    def correctness(self, conversation):
        """Evaluate correctness."""
        criteria = "Correctness"
        criteria_des = """
The correctness of the information provided by the Teaching Assistant is paramount. 
How factually accurate are the things the Teaching Assistant is saying? 
More correct should be given a higher score, and less correct should be given a lower score.
"""
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def correctness_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Correctness", "score": float(self.correctness_output['Grade'])}


    def relevance(self, conversation):
        """Evaluate relevance."""
        criteria = "Relevance"
        criteria_des = """
The Teaching Assistant should deliver content that is directly relevant to the topic being discussed, avoiding tangential information unless it enhances understanding. 
How relevant are the things the Teaching Assistant is saying? 
More relevant should be given a higher score, and less relevant should be given a lower score.
"""
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def relevance_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Relevance", "score": float(self.correctness_output['Grade'])}


    def clarity(self, conversation):
        """Evaluate clarity."""
        criteria = "Clarity"
        criteria_des = """
The ability of the Teaching Assistant to explain concepts clearly and understandably. This includes using appropriate vocabulary and structuring responses in a logical manner. 
How clearly is the Teaching Assistant communicating? 
Clearer communication should be given a higher score, and less clear communication should be given a lower score.
"""
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def clarity_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Clarity", "score": float(self.clarity_output['Grade'])}


    def adaptability(self, conversation):
        """Evaluate adaptability."""
        criteria = "Adaptability"
        criteria_des = """
The ability to adjust explanations or the level of detail based on the student's responses, questions, or level of understanding.
How well is the Teaching Assistant adapting it's teaching approach to the student? 
Better adaptability should be given a higher score, and lack of adaptability should be given a lower score.
"""
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def adaptability_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Adaptability", "score": float(self.adaptability_output['Grade'])}








if __name__ == "__main__":

    examples = "No examples"
    """
CONVERSATION 1:
\nStudent: Hello, I am August!\nI am studying the course IntroToMachineLearning and am trying to learn about the subject Linear Regression.\nMy learning preferences are described as the following: I prefer code examples in order to understand technical concepts.\nPlease explain me this subject.
\nTeaching Assistant: Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. The simplest form of the regression equation with one dependent and one independent variable is defined by the formula $y = \\beta_0 + \\beta_1x$, where:\n- $y$ is the dependent variable,\n- $x$ is the independent variable,\n- $\\beta_0$ is the intercept,\n- $\\beta_1$ is the slope of the line.\n\nThis equation allows you to predict the value of $y$ based on the value of $x$. In machine learning, linear regression can be used for predicting outcomes and trends based on historical data.\n\nHere's a Python code example using scikit-learn to perform linear regression:\n\n```python\nimport numpy as np\nfrom sklearn.linear_model import LinearRegression\nimport matplotlib.pyplot as plt\n\n# Sample data\n# Independent variable\nX = np.array([[1], [2], [3], [4], [5]])\n# Dependent variable\ny = np.array([2, 4, 5, 4, 5])\n\n# Create a linear regression model\nmodel = LinearRegression()\n\n# Fit the model\nmodel.fit(X, y)\n\n# Predict values\npredicted = model.predict(X)\n\n# Plotting the results\nplt.scatter(X, y, color='red', label='Actual data')\nplt.plot(X, predicted, color='blue', label='Fitted line')\nplt.title('Linear Regression Example')\nplt.xlabel('Independent variable (X)')\nplt.ylabel('Dependent variable (y)')\nplt.legend()\nplt.show()\n```\n\nIn this example:\n- We create sample data for $X$ (independent variable) and $y$ (dependent variable).\n- We instantiate the `LinearRegression` model and fit it to our data.\n- We then use the model to predict $y$ values based on $X$.\n- Finally, we plot both the actual data points and the fitted line to visualize how well our model has performed.\n\nThis code will help you see how linear regression works in practice, from setting up the data and model to making predictions and visualizing the results.
\nStudent: I'm not sure I understand the subject from this explanation. Can you explain it in a different way?
\nTeaching Assistant: Let's explore linear regression with a new dataset to enhance your understanding. In this example, we generate a set of 100 random data points for the independent variable $X$, ranging from 0 to 10. We then define the dependent variable $y$ using the equation $y = 3X + \\epsilon + 5$, where $\\epsilon$ is random noise. This helps simulate real-world data where measurements are not perfect.\n\nWe fit a linear regression model to this data, aiming to find the best line that approximates the relationship between $X$ and $y$. The model's predictions are then plotted alongside the actual data points. The green dots represent the actual data, while the black line shows the fitted model's predictions.\n\nThis visual representation helps you see how the linear regression model attempts to draw a line that minimizes the distance between itself and all the data points, effectively capturing the trend in the data. This example should give you a clearer idea of how linear regression works and how it can be applied to different datasets to predict outcomes based on historical data.
\n\n
FEEDBACK FOR CONVERSATION 1:\n
Personalization criteria Score: 1.0\n
Engagement criteria Score: 0.25\n
Repetition criteria Score: 0.75\n
Correctness criteria Score: 1.0\n\n

CONVERSATION:
\nStudent: Hello! I am trying to learn about non-linear regression today. Can you explain this topic?
\nTeaching Assistant: Hello! Nonlinear regression is a valuable method used in regression analysis when data cannot be accurately represented by a straight line. In this type of regression, the relationship between variables is modeled by a function that is a nonlinear combination of model parameters and depends on independent variables.\n\nTo put it simply, nonlinear regression involves finding the best-fitting curve or function that represents the relationship between variables through successive approximations. This method is particularly useful when the relationship between variables is not linear and a straight line would not accurately capture the data.\n\nFor example, in the field of economics, nonlinear regression can be used to model the relationship between income and spending habits, as people may not spend in a linear manner as their income increases.\n\nIt is important to note that non-linear regression can be more complex and computationally intensive compared to linear regression. This complexity arises from the need to estimate parameters for the nonlinear function, which can require advanced mathematical techniques and computational power.\n\nI hope this explanation, along with the example provided, helps you grasp the basics of non-linear regression. Feel free to reach out if you have any further questions or need clarification.
\nStudent: Okay, can you give some formulas that relate to non-linear formulas in economics?
\nTeaching Assitant: Hello! Nonlinear regression is a valuable method used in regression analysis when data cannot be accurately represented by a straight line. In this type of regression, the relationship between variables is modeled by a function that is a nonlinear combination of model parameters and depends on independent variables.\n\nTo put it simply, nonlinear regression involves finding the best-fitting curve or function that represents the relationship between variables through successive approximations. This method is particularly useful when the relationship between variables is not linear and a straight line would not accurately capture the data.\n\nIn economics, non-linear formulas play a crucial role in modeling relationships between variables. For example, polynomial functions like y=w0+w1x+w2x2+w3x3 and trigonometric functions like y=w0+w1cos(x) +w2sin(4x) are commonly used to capture non-linear features in the data. These formulas help economists understand complex relationships that cannot be explained by linear models.\n\nNonlinear regression can be more complex and computationally intensive compared to linear regression. It involves estimating parameters for the nonlinear function using advanced mathematical techniques and computational power. Despite its complexity, nonlinear regression is a powerful tool in analyzing economic data and uncovering non-linear relationships between variables.\n\nI hope this explanation, along with the examples provided, helps you grasp the basics of non-linear regression in economics. If you have any further questions or need clarification, please feel free to reach out.\n\nAdditionally, for further exploration of non-linear regression and non-linear formulas in economics, I recommend checking out resources such as textbooks, online courses, or academic papers. These additional resources can deepen your understanding of the topic and enhance your knowledge in this area.
\n\nFEEDBACK FOR CONVERSATION 2:
\nPersonalization criteria Score: 0.25
\nEngagement criteria Score: 0.25
\nRepetition criteria Score: 0.75
\nCorrectness criteria Score: 0.5
\n\n
"""

    langsmith_name =  "TAS Evaluation"
    dataset_name = "TAS-GPT4_Repetition"
    langsmith_name = langsmith_name + " " + dataset_name
    llm_model = init_llm_langsmith(llm_key=40, temp=0.5, langsmith_name=langsmith_name)

    time_now = time.strftime("%Y.%m.%d-%H.%M.")
    experiment_name = langsmith_name + time_now
    evaluator_class = TasEvaluator(eval_model=llm_model,
                                   experiment_name=experiment_name,
                                   few_shot_examples=examples)

    # dataset_name = "TAS_v1_GPT4_Dataset"
    experiment_results = evaluator_class.run_evaluation(dataset_name=dataset_name)




# class TasEvaluator:
#     """Class for evaluating the Teaching Agent System."""

#     def __init__(self, eval_model, experiment_name: str):
#         """Initialize."""
#         self.eval_model = eval_model
#         self.experiment_name = experiment_name

#     """Prompt for the TAS Evaluator."""
#     def build_eval_prompt(self, prompt_name: str, criteria: str, criteria_des: str):
#         """Build the agent prompt."""
#         prompt_hub_template = hub.pull(prompt_name).template
#         prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
#         prompt = prompt_template.partial(criteria_name=criteria,
#                                          criteria_description=criteria_des)
#         return prompt

#     """Parses the dataset to extract the chat history which is evaluated upon."""
#     def output_dataset(self, inputs: dict) -> dict:
#         """Extract entire chat history from dataset."""
#         chat_hist = inputs["chat_history"]
#         conversation = ""
#         for message in chat_hist:
#             if message['type'] == "ai":
#                 conversation += f"Teaching Assistant: {message['content']}\n\n"
#             else:
#                 conversation += f"Student: {message['content']}\n\n"
#         return {"chat_history": conversation}

#     """Run the evaluation."""
#     def run_evaluation(self, dataset_name):
#         """Run the evaluation experiment."""
#         other_metrics = OtherEvaluationMetrics()
#         evalulators = [self.evaluator_correctness,
#                        self.clarity,
#                        self.funnyness,
#                        self.adaptability,
#                        self.politeness,
#                        other_metrics.is_answered,
#                        other_metrics.conversation_length]
#         # Run
#         experiment_results = evaluate(self.output_dataset,
#                 data=dataset_name,
#                 evaluators=evalulators,
#                 experiment_prefix=self.experiment_name,
#                 # Any experiment metadata can be specified here
#                 # metadata={"variant": "stuff website context into gpt-3.5-turbo",},
#                 )
#         return experiment_results


#     def correctness(self, run: Run, example: Example) -> dict:
#         """Evaluate correctness."""
#         criteria = "Correctness"
#         criteria_des = "How factually correct are the things the teaching assistant is saying? The more factual errors, the lower the score."
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-correctness",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         conversation = run.outputs.get("chat_history")
#         output = chain.invoke({"input": conversation})
#         return {"key": criteria , "score": float(output['Grade'])}

#     def clarity(self, run: Run, example: Example) -> dict:
#         """Evaluate clarity."""
#         criteria = "Clarity"
#         criteria_des = "How clearly is the teaching assistant communicating? The more clear, the higher the score."
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-correctness",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         conversation = run.outputs.get("chat_history")
#         output = chain.invoke({"input": conversation})
#         return {"key": criteria , "score": float(output['Grade'])}

#     def funnyness(self, run: Run, example: Example) -> dict:
#         """Evaluate funnyness."""
#         criteria = "Funnyness"
#         criteria_des = "How funny is teaching assistant, and is it making any jokes? The more funny, the higher the score."
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-correctness",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         conversation = run.outputs.get("chat_history")
#         output = chain.invoke({"input": conversation})
#         return {"key": criteria , "score": float(output['Grade'])}

#     def adaptability(self, run: Run, example: Example) -> dict:
#         """Evaluate adaptability."""
#         criteria = "Student Adaptability"
#         criteria_des = "How well is the teaching assistant adapting it's teaching approach to the student? The more adaptable, the higher the score."
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-correctness",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         conversation = run.outputs.get("chat_history")
#         output = chain.invoke({"input": conversation})
#         return {"key": criteria , "score": float(output['Grade'])}

#     def politeness(self, run: Run, example: Example) -> dict:
#         """Evaluate politeness."""
#         criteria = "Politeness"
#         criteria_des = "How polite is the teaching assistant? The more polite, the higher the score."
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-correctness",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         conversation = run.outputs.get("chat_history")
#         output = chain.invoke({"input": conversation})
#         return {"key": criteria , "score": float(output['Grade'])}





# class TasEvaluator:
#     """Class for evaluating the Teaching Agent System."""

#     def __init__(self, eval_model, experiment_name: str):
#         """Initialize."""
#         self.eval_model = eval_model
#         self.experiment_name = experiment_name
#         self.eval_prompt = self.build_eval_prompt()
#         self.eval_chain = self.build_eval_chain()

#     """Parses the dataset to extract the chat history which is evaluated upon."""
#     def output_dataset(self, inputs: dict) -> dict:
#         """Extract entire chat history from dataset."""
#         chat_hist = inputs["chat_history"]
#         chat_string = ""
#         for message in chat_hist:
#             if message['type'] == "ai":
#                 chat_string += f"Teaching Assistant: {message['content']}\n\n"
#             else:
#                 chat_string += f"Student: {message['content']}\n\n"
#         return {"chat_history": chat_string}

#     """Prompt for the TAS Evaluator."""
#     def build_eval_prompt(self, prompt_name: str):
#         """Build the agent prompt."""
#         criteria = """Correctness: Correctness of the answer.\nCriteria 2: Clarity of the answer."""#\nCriteria 3: Relevance of the answer.\nCriteria 4: Engagement with the student."""

#         prompt_hub_template = hub.pull(prompt_name).template
#         prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
#         prompt = prompt_template.partial(criteria=criteria)
#         return prompt

#     def build_eval_chain(self):
#         """Build the evaluation chain."""
#         chain = self.eval_prompt | self.eval_model | JsonOutputParser()#output_parser.StrOutputParser()
#         return chain

#     def run_eval_chain(self, run: Run, example: Example) -> EvaluationResults:
#         """Run the evaluation chain."""
#         conversation = run.outputs.get("chat_history")
#         output = self.eval_chain.invoke({"input": conversation})


#         print("\nResponse:\n", output)
#         criteria_grades = {}
#         criterias = []
#         grades = []
#         for i, criteria in enumerate(output):
#             print("\nCriteria:\n", criteria)
#             for key, value in criteria.items():
#                 criteria_grade = {"key": str(key) , "score": float(value['Grade'])}
#                 criterias.append(key)
#                 grades.append(float(value['Grade']))
#                 # criteria_grades.append(criteria_grade)
#                 criteria_grades[f"Criteria {i+1}"] = criteria_grade
#                 # self.return_grade(criteria_grade)
#         print("\nCriteria grades:\n", criteria_grades)


#     def correctness(self, run: Run, example: Example) -> dict:

#         "augustsemrau/react-teaching-chat-evaluator"



#     def run_experiment(self, dataset_name):
#         """Run the evaluation experiment."""
#         other_metrics = OtherEvaluationMetrics()
#         evalulators = [self.run_eval_chain,
#                        other_metrics.is_answered,
#                        other_metrics.conversation_length]
#         # Run
#         experiment_results = evaluate(self.output_dataset,
#                 data=dataset_name,
#                 evaluators=evalulators,
#                 experiment_prefix=self.experiment_name,
#                 # Any experiment metadata can be specified here
#                 # metadata={"variant": "stuff website context into gpt-3.5-turbo",},
#                 )
#         return experiment_results

