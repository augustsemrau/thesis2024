"""Module contains the implementation of the TAS evaluation."""

# Langchain imports
from langchain import chat_models, prompts, smith, hub
from langchain.schema import output_parser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Langsmith imports
from langsmith.evaluation import evaluate, EvaluationResult, EvaluationResults
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
    """Class for evaluating the Teaching Agent System."""

    def __init__(self, eval_model, experiment_name: str):
        """Initialize."""
        self.eval_model = eval_model
        self.experiment_name = experiment_name


    """Prompt for the TAS Evaluator."""
    def build_eval_prompt(self, prompt_name: str, criteria: str, criteria_des: str):
        """Build the agent prompt."""
        prompt_hub_template = hub.pull(prompt_name).template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial(criteria_name=criteria,
                                         criteria_description=criteria_des)
        return prompt

    """Parses the dataset to extract the chat history which is evaluated upon."""
    def output_dataset(self, inputs: dict) -> dict:
        """Extract entire chat history from dataset."""
        chat_hist = inputs["chat_history"]
        conversation = ""
        for message in chat_hist:
            if message['type'] == "ai":
                conversation += f"Teaching Assistant: {message['content']}\n\n"
            else:
                conversation += f"Student: {message['content']}\n\n"
        self.correctness_output = self.correctness(conversation=conversation)
        self.clarity_output = self.clarity(conversation=conversation)
        self.funnyness_output = self.funnyness(conversation=conversation)
        self.adaptability_output = self.adaptability(conversation=conversation)
        self.politeness_output = self.politeness(conversation=conversation)

        return {"Correctness": self.correctness_output,
                "Clarity": self.clarity_output,
                "Funnyness": self.funnyness_output,
                "Adaptability": self.adaptability_output,
                "Politeness": self.politeness_output,
                "Conversation": conversation}

    """Run the evaluation."""
    def run_evaluation(self, dataset_name):
        """Run the evaluation experiment."""
        other_metrics = OtherEvaluationMetrics()
        evalulators = [self.correctness_grade,
                       self.clarity_grade,
                       self.funnyness_grade,
                       self.adaptability_grade,
                       self.politeness_grade,
                       other_metrics.is_answered,
                       other_metrics.conversation_length]
        # Run
        experiment_results = evaluate(self.output_dataset,
                data=dataset_name,
                evaluators=evalulators,
                experiment_prefix=self.experiment_name,
                # Any experiment metadata can be specified here
                # metadata={"variant": "stuff website context into gpt-3.5-turbo",},
                )
        return experiment_results

    """Evaluate correctness."""
    def correctness(self, conversation):
        """Evaluate correctness."""
        criteria = "Correctness"
        criteria_des = "How factually correct are the things the teaching assistant is saying? The more factual errors, the lower the score."
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-correctness",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def correctness_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Correctness", "score": float(self.correctness_output['Grade'])}

    """Evaluate clarity."""
    def clarity(self, conversation):
        """Evaluate clarity."""
        criteria = "Clarity"
        criteria_des = "How clearly is the teaching assistant communicating? The more clear, the higher the score."
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-correctness",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def clarity_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Clarity", "score": float(self.clarity_output['Grade'])}

    """Evaluate funnyness."""
    def funnyness(self, conversation):
        """Evaluate funnyness."""
        criteria = "Funnyness"
        criteria_des = "How funny is teaching assistant, and is it making any jokes? The more funny, the higher the score."
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-correctness",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def funnyness_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Funnyness", "score": float(self.funnyness_output['Grade'])}

    """Evaluate adaptability."""
    def adaptability(self, conversation):
        """Evaluate adaptability."""
        criteria = "Teaching Adaptability"
        criteria_des = "How well is the teaching assistant adapting it's teaching approach to the student? The more adaptable, the higher the score."
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-correctness",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def adaptability_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Teaching Adaptability", "score": float(self.adaptability_output['Grade'])}

    """Evaluate politeness."""
    def politeness(self, conversation):
        """Evaluate politeness."""
        criteria = "Politeness"
        criteria_des = "How polite is the teaching assistant? The more polite, the higher the score."
        prompt = self.build_eval_prompt(prompt_name="augustsemrau/tas-evaluator-correctness",
                                        criteria=criteria,
                                        criteria_des=criteria_des)
        chain = prompt | self.eval_model | JsonOutputParser()
        return chain.invoke({"input": conversation})
    def politeness_grade(self, run: Run, example: Example) -> dict:
        """Output correctness grade to Langsmith."""
        return {"key": "Politeness", "score": float(self.politeness_output['Grade'])}





if __name__ == "__main__":
    langsmith_name = "Langsmith Eval Experiment 1"
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name=langsmith_name)

    experiment_name = "TAS Evaluation Test 1"
    evaluator_class = TasEvaluator(eval_model=llm_model, experiment_name=experiment_name)

    dataset_name = "TAS v1 Conversations"
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

