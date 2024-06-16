"""Module contains the implementation of the TAS evaluation."""

import time

# Langchain imports
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Langsmith imports
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

# Local imports
from thesis2024.utils import init_llm_langsmith


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
        """Redundant."""
        chat_hist = inputs["chat_history"]
        conversation = str(chat_hist)
        return {"All"
                "Conversation": conversation}


    def run_evaluation(self, dataset_name):
        """Run the evaluation experiment."""
        experiment_results = evaluate(
                self.output_dataset,
                data=dataset_name,
                evaluators=[self.eval],
                experiment_prefix=self.experiment_name,
                num_repetitions=3,
                # max_concurrency=1,
                # Any experiment metadata can be specified here
                # metadata={"variant": "stuff website context into gpt-3.5-turbo",},
                )
        return experiment_results


    def eval(self, run: Run, example: Example) -> dict:
        """Evaluate correctness."""
        criteria = "- Personalization\n- Engagement\n- Repetition"
        criteria_description = """
Personalization Description:\n
The student states their personal learning preferences, and the Teaching Assistant should adapt to these preferences. Adapting it's explanations to these learning preferences is paramount. 
How well does the Teaching Assistant personalize it's explanation of the given subject to the student? 
Better personalization be given a higher score, and neglecting the student's learning preferences should be given a lower score.
\n\nEngagement Description:\n
The teaching assistant should engage the student in the conversation, making it interesting and interactive. Asking questions, providing examples, and encouraging participation are all ways to increase engagement.
How engaging is the Teaching Assistant in the conversation?
More engaging should be given a higher score, and less engaging should be given a lower score.
\n\nRepetition Description:\n
Repeating the same information multiple times can be necessary in order for the student to understand it, but should be conveyed in alternative ways by rephrasing and re-contextualising.
How well is the Teaching Assistant conveying the information without repeating itself?
Avoiding the repeating of information should be given a higher score. If repeats are done, the ability to rephrase the same information in new ways should be given a higher score. Repeating information without altering it should be given a lower score.
"""
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-3criteria",
                                        criteria=criteria,
                                        criteria_des=criteria_description)
        chain = prompt | self.eval_model | JsonOutputParser()

        conversation = example.inputs["chat_history"]
        evaluation = chain.invoke({"input": conversation})
        self.eval_output = evaluation
        return {
        "results": [
            # Provide the key, score and other relevant information for each metric
            {"key": "Personalization", "score": evaluation['Personalization']['Personalization Grade'], "feedback": evaluation['Personalization']['Personalization Feedback']},
            {"key": "Engagement", "score": evaluation['Engagement']['Engagement Grade'], "feedback": evaluation['Engagement']['Engagement Feedback']},
            {"key": "Repetition", "score": evaluation['Repetition']['Repetition Grade'], "feedback": evaluation['Repetition']['Repetition Feedback']},
            {"key": "conversation_length" , "score": len(conversation)}]}



if __name__ == "__main__":

    examples = """
CONVERSATION A:
\n
"Student: Hello, I am August!\nI am studying the course IntroToMachineLearning and am trying to learn about the subject Linear Regression.\nMy learning preferences are described as the following: I prefer formulas and math in order to understand technical concepts.\nPlease explain me this subject.
\nTeaching Assistant: Linear regression involves fitting a regression model to data points to predict a target variable. In the case of a 1D dataset, feature transformations are applied to the data to fit models such as plain linear regression or second-degree polynomial regression. The weight vector is computed based on these transformations to make predictions for new data points.
\nStudent: I'm not sure I understand the subject from this explanation. Can you explain it in a different way?
\nTeaching Assistant: Linear regression involves fitting a regression model to data points to predict a target variable. In simple terms, we use mathematical formulas to find the best-fitting line or curve that represents the relationship between the input variables (features) and the output variable (target). By using techniques like feature transformations and computing weight vectors, we can make predictions for new data points. The key formula involved in linear regression is $w^* = (X^TX)^{-1}X^Ty$, where $w^*$ is the weight vector, X is the feature matrix, and y is the target variable. This formula helps us find the optimal weights to make accurate predictions."
\n\n
EVALUATION FOR CONVERSATION A:\n
Personalization Feedback: The Teaching Assistant initially does not use any math, and explains the subject using technical terms. Only when asked to explain the subject in a different way, does it use a little formulaic explanation, however this is not good.\n
Personalization Score: 0.25\n
Engagement Feedback: The Teaching Assistant does not engage the student in any way, and the explanations it gives are short and boiled down, which does not allow for engagement from the student's side.\n
Engagement criteria Score: 0.0\n
Repetition Feedback: In the second answer, the Teaching Assistant actually takes a different approach than in the first, and uses some math. However, it essentially repeats sentences such as the one about computing weights, which is undesirable.\n
Repetition Score: 0.25\n
\n\n\n
CONVERSATION B:
\n
Student: Hello, I am August!\nI am studying the course IntroToMachineLearning and am trying to learn about the subject Linear Regression.\nMy learning preferences are described as the following: I prefer formulas and math in order to understand technical concepts.\nPlease explain me this subject.
\nTeaching Assistant: Linear regression is a method used to model the relationship between a dependent variable $y$ and one or more independent variables $x$. In the simplest form, known as simple linear regression, the relationship is modeled as $y = mx + b$, where $m$ is the slope of the line and $b$ is the y-intercept. The goal is to find the values of $m$ and $b$ that best fit the data points.\n\nTo generalize to multiple independent variables, we can use multiple linear regression. In this case, the relationship is modeled as $y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$, where $w_0, w_1, w_2, ..., w_n$ are the coefficients to be determined.\n\nTo find the optimal values of the coefficients, we use a method called least squares, which minimizes the sum of the squared differences between the observed values and the values predicted by the model. This is typically done using matrix operations.\n\nIn summary, linear regression is a powerful tool for modeling relationships between variables and making predictions based on data points. It involves finding the best-fitting line or plane that represents the relationship between the variables.
\nStudent: I'm not sure I understand the subject from this explanation. Can you explain it in a different way?
\nTeaching Assistant: Linear regression involves fitting a model to data points to make predictions. In simple linear regression, the relationship between a dependent variable $y$ and an independent variable $x$ is modeled as $y = mx + b$, where $m$ is the slope and $b$ is the y-intercept. The goal is to find the best values of $m$ and $b$ that fit the data. For multiple linear regression with multiple independent variables, the relationship is modeled as $y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$, where $w_0, w_1, w_2, ..., w_n$ are coefficients to be determined. The optimal coefficients are found using the method of least squares, which minimizes the sum of squared differences between observed and predicted values. Linear regression is a method for modeling relationships between variables and making predictions based on data points by finding the best-fitting line or plane that represents the relationship between the variables.
\n\n
EVALUATION FOR CONVERSATION B:\n
Personalization Feedback: The Teaching Assistant initially utilizes formulas to explain the subject, in accordance to the student's learning preferences. However, the math it writes is rather simple.\n
Personalization Score: 0.50\n
Engagement Feedback: The Teaching Assistant does not engage the student in any meaningful way.\n
Engagement criteria Score: 0.0\n
Repetition Feedback: In the second answer, the Teaching Assistant reuses the exact same explanation of the subject, which is not good.\n
Repetition Score: 0.0\n
\n\n\n
CONVERSATION C:
\n
"Student: Hello, I am August!\nI am studying the course IntroToMachineLearning and am trying to learn about the subject Linear Regression.\nMy learning preferences are described as the following: I prefer formulas and math in order to understand technical concepts.\nPlease explain me this subject.
\nTeaching Assistant: Linear regression models the relationship between a dependent variable $y$ and one or more independent variables $x$. The simplest form is simple linear regression, where there is only one independent variable. The model is represented by the equation:\n\n$$y = \\beta_0 + \\beta_1x + \\epsilon$$\n\nHere:\n- $y$ is the dependent variable.\n- $x$ is the independent variable.\n- $\\beta_0$ and $\\beta_1$ are coefficients that represent the intercept and the slope of the regression line, respectively.\n- $\\epsilon$ represents the error term, which accounts for the variability in $y$ that cannot be explained by $x$ alone.\n\nThe coefficients $\\beta_0$ and $\\beta_1$ are determined using the method of least squares. This method minimizes the sum of the squares of the residuals (the differences between observed values and the values predicted by the model). Mathematically, this is expressed as:\n\n$$\\min_{\\beta_0, \\beta_1} \\sum_{i=1}^n (y_i - (\\beta_0 + \\beta_1 x_i))^2$$\n\nWhere:\n- $y_i$ is the observed value of the dependent variable for the $i$-th observation.\n- $x_i$ is the value of the independent variable for the $i$-th observation.\n- $n$ is the number of observations.\n\nThis formulation allows us to estimate the parameters of the model that best fit the data, providing a predictive equation that can be used to estimate $y$ based on new values of $x$.\n\nIf you have data and want to see this model applied, or if you need further explanation on any of the terms or processes, feel free to ask!
\nStudent: I'm not sure I understand the subject from this explanation. Can you explain it in a different way?
\nTeaching Assistant: Certainly, August! Let's explore linear regression with a focus on the derivation of the least squares method, which helps us find the best-fitting line for the given data.\n\n### Step 1: Define the Error\nThe error for each data point is the difference between the observed value $y_i$ and the predicted value $\\hat{y}_i$. The predicted value is given by the linear model:\n\n$$\\hat{y}_i = \\beta_0 + \\beta_1 x_i$$\n\n### Step 2: Sum of Squared Errors\nThe goal is to find the values of $\\beta_0$ and $\\beta_1$ that minimize the sum of the squared errors (SSE). The SSE is defined as:\n\n$$SSE = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 = \\sum_{i=1}^n (y_i - (\\beta_0 + \\beta_1 x_i))^2$$\n\n### Step 3: Minimize SSE\nTo find the minimum SSE, we take the partial derivatives of SSE with respect to $\\beta_0$ and $\\beta_1$ and set them to zero.\n\n#### Derivative with respect to $\\beta_0$:\n$$\\frac{\\partial}{\\partial \\beta_0} SSE = -2 \\sum_{i=1}^n (y_i - \\beta_0 - \\beta_1 x_i) = 0$$\n\n#### Derivative with respect to $\\beta_1$:\n$$\\frac{\\partial}{\\partial \\beta_1} SSE = -2 \\sum_{i=1}^n x_i(y_i - \\beta_0 - \\beta_1 x_i) = 0$$\n\n### Step 4: Solve the Equations\nFrom the first equation, we can express $\\beta_0$:\n$$n \\beta_0 + \\beta_1 \\sum_{i=1}^n x_i = \\sum_{i=1}^n y_i$$\n$$\\beta_0 = \\frac{\\sum_{i=1}^n y_i - \\beta_1 \\sum_{i=1}^n x_i}{n}$$\n\nPlugging $\\beta_0$ into the second equation, we can solve for $\\beta_1$:\n$$\\beta_1 = \\frac{n \\sum_{i=1}^n x_i y_i - \\sum_{i=1}^n x_i \\sum_{i=1}^n y_i}{n \\sum_{i=1}^n x_i^2 - (\\sum_{i=1}^n x_i)^2}$$\n\n### Step 5: Interpretation\nThe values of $\\beta_0$ and $\\beta_1$ that satisfy these equations will minimize the SSE, providing the best fit line for the data. $\\beta_0$ is the intercept of the line, and $\\beta_1$ is the slope. This line can then be used to predict new values of $y$ given values of $x$.\n\nThis mathematical approach ensures that the linear regression model is the best possible linear approximation for the given data according to the least squares criterion. If you need further clarification or examples, feel free to ask!"
\n\n
EVALUATION FOR CONVERSATION C:\n
Personalization Feedback: The Teaching Assistant uses extensive math-based explanations, and writes several important formulas for the subject.\n
Personalization Score: 0.75\n
Engagement Feedback: The Teaching Assistant ends each response with engaging the student, which is good. I could however, do this more elaborately.\n
Engagement criteria Score: 0.50\n
Repetition Feedback: The Teaching Assistant takes a completely different approach when re-explaining the subject, while still using relevant topics.\n
Repetition Score: 0.75\n
"""
    """
    \n\n\n
    CONVERSATION D:

    \n\n
    EVALUATION FOR CONVERSATION D:\n
    Personalization Feedback: \n
    Personalization Score: 0.\n
    Engagement Feedback: \n
    Engagement criteria Score: 0.\n
    Repetition Feedback: \n
    Repetition Score: 0.\n
    \n\n\n
    CONVERSATION E:

    \n\n
    EVALUATION FOR CONVERSATION E:\n
    Personalization Feedback: \n
    Personalization Score: 0.\n
    Engagement Feedback: \n
    Engagement criteria Score: 0.\n
    Repetition Feedback: \n
    Repetition Score: 0.\n
    """
    # examples = ""
    langsmith_name =  "TAS_Eval_Dataset_"
    # dataset_name = "TAS_GPT-4_EvaluationSet_1"
    # dataset_name = "TAS_GPT-3.5_EvaluationSet_1"
    # dataset_name = "Eval_FewShotExamples"
    # dataset_name = "BASELINE_GPT-4"
    # dataset_name = "Eval_Testing"
    # dataset_name = "Reflexion_EvaluationSet_1"
    dataset_name = "TMAS_Iterations_0"
    # dataset_name = "TMAS_Iterations_1"
    # dataset_name = "TMAS_Iterations_2"
    # dataset_name = "TMAS_Iterations_3"
    # dataset_name = "TAS_GPT35_Final"
    # dataset_name = "TAS_GPT4_Final"
    langsmith_name = langsmith_name + dataset_name
    llm_model = init_llm_langsmith(llm_key=40, temp=0.5, langsmith_name=langsmith_name)

    time_now = time.strftime("%Y.%m.%d-%H.%M.")
    experiment_name = langsmith_name + time_now
    evaluator_class = TasEvaluator(eval_model=llm_model,
                                   experiment_name=experiment_name,
                                   few_shot_examples=examples)

    # dataset_name = "TAS_v1_GPT4_Dataset"
    experiment_results = evaluator_class.run_evaluation(dataset_name=dataset_name)



# class OtherEvaluationMetrics:
#     """Class for other evaluation metrics."""

#     def __init__(self):
#         """Initialize."""
#         pass

#     def is_answered(self, run: Run, example: Example) -> dict:
#         """Check if the question is answered."""
#         conversation = run.outputs.get("Conversation")
#         if not conversation:
#             return {"key": "is_answered" , "score": 0}
#         else:
#             return {"key": "is_answered" , "score": 1}

#     def conversation_length(self, run: Run, example: Example) -> dict:
#         """Check the length of the conversation."""
#         conversation = run.outputs.get("Conversation")
#         return {"key": "conversation_length" , "score": len(conversation)}




# class TasEvaluator:
#     """Class for evaluating the TAS."""

#     def __init__(self, eval_model, experiment_name: str, few_shot_examples: str):
#         """Initialize."""
#         self.eval_model = eval_model
#         self.experiment_name = experiment_name
#         self.few_shot_examples = few_shot_examples

#     def build_eval_prompt(self, prompt_name: str, criteria: str, criteria_des: str):
#         """Prompt for the TAS Evaluator."""
#         prompt_hub_template = hub.pull(prompt_name).template
#         prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
#         prompt = prompt_template.partial(criteria_name=criteria,
#                                          criteria_description=criteria_des,
#                                          few_shot_examples=self.few_shot_examples)
#         return prompt

#     def output_dataset(self, inputs: dict) -> dict:
#         """Parse the dataset to extract the chat history which is evaluated upon."""
#         chat_hist = inputs["chat_history"]
#         conversation = str(chat_hist) #+ "\n" + second_last_message
#         self.eval_output = self.eval(conversation=conversation)
#         # self.personalization_output = self.personalization(conversation=conversation)
#         # self.engagement_output = self.engagement(conversation=conversation)
#         # self.repetition_output = self.repetition(conversation=conversation)
#         return {"All"
#                 "Evaluation": self.eval_output,
#                 # "Personalization": self.personalization_output,
#                 # "Engagement": self.engagement_output,
#                 # "Repetition": self.repetition_output,
#                 "Conversation": conversation}


#     def run_evaluation(self, dataset_name):
#         """Run the evaluation experiment."""
#         other_metrics = OtherEvaluationMetrics()
#         evalulators = [other_metrics.is_answered,
#                        other_metrics.conversation_length,
#                        self.eval_personalization_grade,
#                        self.eval_engagement_grade,
#                        self.eval_repetition_grade,
#                     #    self.personalization_grade,
#                     #    self.engagement_grade,
#                     #    self.repetition_grade,
#                        ]
#         # Run
#         experiment_results = evaluate(self.output_dataset,
#                 data=dataset_name,
#                 evaluators=evalulators,
#                 experiment_prefix=self.experiment_name,
#                 # max_concurrency=1,
#                 # Any experiment metadata can be specified here
#                 # metadata={"variant": "stuff website context into gpt-3.5-turbo",},
#                 )
#         return experiment_results


#     def eval(self, conversation):
#         """Evaluate correctness."""
#         criteria = "- Personalization\n- Engagement\n- Repetition"
#         criteria_description = """
# Personalization Description:\n
# The student states their personal learning preferences, and the Teaching Assistant should adapt to these preferences. Adapting it's explanations to these learning preferences is paramount. 
# How well does the Teaching Assistant personalize it's explanation of the given subject to the student? 
# Better personalization be given a higher score, and neglegting the student's learning preferences should be given a lower score.
# \n\nEngagement Description:\n
# The teaching assistant should engage the student in the conversation, making it interesting and interactive. Asking questions, providing examples, and encouraging participation are all ways to increase engagement.
# How engaging is the Teaching Assistant in the conversation?
# More engaging should be given a higher score, and less engaging should be given a lower score.
# \n\nRepetition Description:\n
# Repeating the same information multiple times can be necessary in order for the student to understand it, but should be conveyed in alternative ways by rephrasing and recontextualing.
# How well is the Teaching Assistant conveying the information without repeating itself?
# Avoiding the repeating of information should be given a higher score. If repeats are done, the ability to rephrase the same information in new ways should be given a higher score. Repeating information without altering it should be given a lower score.
# """
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-3criteria",
#                                         criteria=criteria,
#                                         criteria_des=criteria_description)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         return chain.invoke({"input": conversation})

#     def eval_personalization_grade(self, run: Run, example: Example) -> dict:
#         """Output personalization grade to Langsmith."""
#         return {"key": "Personalization", "score": float(self.eval_output['Personalization']['Personalization Grade'])}
#     def eval_engagement_grade(self, run: Run, example: Example) -> dict:
#         """Output engagement grade to Langsmith."""
#         return {"key": "Engagement", "score": float(self.eval_output['Engagement']['Engagement Grade'])}, {"key": "Repetition", "score": float(self.eval_output['Repetition']['Repetition Grade'])}
#     def eval_repetition_grade(self, run: Run, example: Example) -> dict:
#         """Output repetition grade to Langsmith."""
#         return {"key": "Repetition", "score": float(self.eval_output['Repetition']['Repetition Grade'])}




#     def personalization(self, conversation):
#         """Evaluate correctness."""
#         criteria = "Personalization"
#         criteria_des = """
# The student states their personal learning preferences, and the Teaching Assistant should adapt to these preferences. Adapting it's explanations to these learning preferences is paramount. 
# How well does the Teaching Assistant personalize it's explanation of the given subject to the student? 
# Better personalization be given a higher score, and neglegting the student's learning preferences should be given a lower score.
# """
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         return chain.invoke({"input": conversation})
#     def personalization_grade(self, run: Run, example: Example) -> dict:
#         """Output correctness grade to Langsmith."""
#         return {"key": "Personalization", "score": float(self.personalization_output['Grade'])}

#     def engagement(self, conversation):
#         """Evaluate correctness."""
#         criteria = "Engagement"
#         criteria_des = """
# The teaching assistant should engage the student in the conversation, making it interesting and interactive. Asking questions, providing examples, and encouraging participation are all ways to increase engagement.
# How engaging is the Teaching Assistant in the conversation?
# More engaging should be given a higher score, and less engaging should be given a lower score.
# """
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         return chain.invoke({"input": conversation})
#     def engagement_grade(self, run: Run, example: Example) -> dict:
#         """Output correctness grade to Langsmith."""
#         return {"key": "Engagement", "score": float(self.engagement_output['Grade'])}

#     def repetition(self, conversation):
#         """Evaluate correctness."""
#         criteria = "Repetition"
#         criteria_des = """
# Repeating the same information multiple times can be necessary in order for the student to understand it, but should be conveyed in alternative ways by rephrasing and recontextualing.
# How well is the Teaching Assistant conveying the information without repeating itself?
# Avoiding the repeating of information should be given a higher score. If repeats are done, the ability to rephrase the same information in new ways should be given a higher score. Repeating information without altering it should be given a lower score.
# """
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         return chain.invoke({"input": conversation})
#     def repetition_grade(self, run: Run, example: Example) -> dict:
#         """Output repetition grade to Langsmith."""
#         return {"key": "Repetition", "score": float(self.repetition_output['Grade'])}







#     def correctness(self, conversation):
#         """Evaluate correctness."""
#         criteria = "Correctness"
#         criteria_des = """
# The correctness of the information provided by the Teaching Assistant is paramount. 
# How factually accurate are the things the Teaching Assistant is saying? 
# More correct should be given a higher score, and less correct should be given a lower score.
# """
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         return chain.invoke({"input": conversation})
#     def correctness_grade(self, run: Run, example: Example) -> dict:
#         """Output correctness grade to Langsmith."""
#         return {"key": "Correctness", "score": float(self.correctness_output['Grade'])}

#     def relevance(self, conversation):
#         """Evaluate relevance."""
#         criteria = "Relevance"
#         criteria_des = """
# The Teaching Assistant should deliver content that is directly relevant to the topic being discussed, avoiding tangential information unless it enhances understanding. 
# How relevant are the things the Teaching Assistant is saying? 
# More relevant should be given a higher score, and less relevant should be given a lower score.
# """
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         return chain.invoke({"input": conversation})
#     def relevance_grade(self, run: Run, example: Example) -> dict:
#         """Output correctness grade to Langsmith."""
#         return {"key": "Relevance", "score": float(self.correctness_output['Grade'])}

#     def clarity(self, conversation):
#         """Evaluate clarity."""
#         criteria = "Clarity"
#         criteria_des = """
# The ability of the Teaching Assistant to explain concepts clearly and understandably. This includes using appropriate vocabulary and structuring responses in a logical manner. 
# How clearly is the Teaching Assistant communicating? 
# Clearer communication should be given a higher score, and less clear communication should be given a lower score.
# """
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         return chain.invoke({"input": conversation})
#     def clarity_grade(self, run: Run, example: Example) -> dict:
#         """Output correctness grade to Langsmith."""
#         return {"key": "Clarity", "score": float(self.clarity_output['Grade'])}

#     def adaptability(self, conversation):
#         """Evaluate adaptability."""
#         criteria = "Adaptability"
#         criteria_des = """
# The ability to adjust explanations or the level of detail based on the student's responses, questions, or level of understanding.
# How well is the Teaching Assistant adapting it's teaching approach to the student? 
# Better adaptability should be given a higher score, and lack of adaptability should be given a lower score.
# """
#         prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-1criteria",
#                                         criteria=criteria,
#                                         criteria_des=criteria_des)
#         chain = prompt | self.eval_model | JsonOutputParser()
#         return chain.invoke({"input": conversation})
#     def adaptability_grade(self, run: Run, example: Example) -> dict:
#         """Output correctness grade to Langsmith."""
#         return {"key": "Adaptability", "score": float(self.adaptability_output['Grade'])}




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

