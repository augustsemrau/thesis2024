"""Module contains the implementation of the TAS evaluation."""

# Langchain imports
from langchain import chat_models, prompts, smith, hub
from langchain.schema import output_parser
from langchain.prompts import PromptTemplate

# Langsmith imports
import langsmith
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from langsmith.wrappers import wrap_openai

import openai

from thesis2024.utils import init_llm_langsmith

import uuid






uid = uuid.uuid4()
client = langsmith.Client()


class OtherEvaluationMetrics:
    """Class for other evaluation metrics."""

    def __init__(self):
        """Initialize."""
        pass

    def is_answered(self, run: Run, example: Example) -> dict:
        """Check if the question is answered."""
        conversation = run.outputs.get("chat_history")
        if not conversation:
            return {"key": "is_answered" , "score": 0}
        else:
            return {"key": "is_answered" , "score": 1}

    def conversation_length(self, run: Run, example: Example) -> dict:
        """Check the length of the conversation."""
        conversation = run.outputs.get("chat_history")
        return {"key": "conversation_lengthd" , "score": len(conversation)}

class TasEvaluator:
    """Class for evaluating the Teaching Agent System."""

    def __init__(self, eval_model):
        """Initialize."""
        self.eval_model = eval_model
        self.eval_prompt = self.build_eval_prompt()
        self.eval_chain = self.build_eval_chain()


    def output_dataset(self, inputs: dict) -> dict:
        """Extract entire chat history from dataset."""
        chat_hist = inputs["chat_history"]
        chat_string = ""
        for message in chat_hist:
            if message['type'] == "ai":
                chat_string += f"Teaching Assistant: {message['content']}\n\n"
            else:
                chat_string += f"Student: {message['content']}\n\n"
        return {"chat_history": chat_string}


    """Prompt for the Teaching Agent System."""
    def build_eval_prompt(self):
        """Build the agent prompt."""
        criteria = """Criteria 1: Correctness of the answer.\nCriteria 2: Clarity of the answer.\nCriteria 3: Relevance of the answer.\nCriteria 4: Engagement with the student."""

        prompt_hub_template = hub.pull("augustsemrau/react-teaching-chat-evaluator").template
        prompt_template = PromptTemplate.from_template(template=prompt_hub_template)
        prompt = prompt_template.partial(criteriae=criteria)
        return prompt

    def build_eval_chain(self):
        """Build the evaluation chain."""
        chain = self.eval_prompt | self.eval_model | output_parser.StrOutputParser()
        return chain

    def run_eval_chain(self, run: Run, example: Example) -> dict:
        """Run the evaluation chain."""
        conversation = run.outputs.get("chat_history")
        response = self.eval_chain.invoke({"input": conversation})
        return response

    def run_experiment(self, dataset_name):
        """Run the evaluation experiment."""
        other_metrics = OtherEvaluationMetrics()
        evalulators = [self.run_eval_chain,
                       other_metrics.is_answered,
                       other_metrics.conversation_length]
        # Run
        experiment_results = evaluate(self.output_dataset,
                data=dataset_name,
                evaluators=evalulators,
                experiment_prefix="test-dbrx-qa-custom-eval-is-answered",
                # Any experiment metadata can be specified here
                metadata={"variant": "stuff website context into gpt-3.5-turbo",},
                )
        return experiment_results




if __name__ == "__main__":
    langsmith_name = None#"Langsmith Eval Experiment 1"
    llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name=langsmith_name)
    evaluator_class = TasEvaluator(eval_model=llm_model)
