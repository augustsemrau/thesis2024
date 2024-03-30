"""Bernchmarking reasoning performance using the ARC dataset."""


import csv

import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def keep_only_alphanumeric(s):
    """Keep only alphanumeric characters."""
    return re.sub(r'\W+', '', s)

def f1_score(prediction, ground_truth):
    """Compute the F1 score of the prediction with respect to the ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class ArcBenhmark:
    """ARC Dataset Benchmarking class."""

    def __init__(self,
                # llm_system,
                dataset_path: str,
                arc_difficulty: str = "easy",
                arc_type: str = "dev",
                ):
        """Initialize the ARC dataset benchmarking class.

        Args:
        ----
        llm_system: object
            The LLM system to be evaluated.
        dataset_path: str
            The path to the ARC dataset.
        arc_difficulty: str, default="easy"
            The difficulty of the dataset. Can be "easy" or "challenge".
        arc_type: str, default="dev"
            The type of the dataset. Can be "dev", "test", or "train".

        """
        # self.llm_system = llm_system
        self.dataset_path = dataset_path
        self.questions = []
        self.load_dataset(difficulty=arc_difficulty, type=arc_type)

    def load_dataset(self, difficulty: str = "easy", type: str = "dev"):
        """Load the dataset.

        Args:
        ----
        difficulty: str, default="easy"
            The difficulty of the dataset. Can be "easy" or "challenge".
        type: str, default="dev"
            The type of the dataset. Can be "dev", "test", or "train".

        """
        if difficulty == "easy":
            if type == "dev":
                self.dataset_path = self.dataset_path + "ARC-Easy/Arc-Easy-Dev.csv"
            elif type == "test":
                self.dataset_path = self.dataset_path + "ARC-Easy/Arc-Easy-Test.csv"
            elif type == "train":
                self.dataset_path = self.dataset_path + "ARC-Easy/Arc-Easy-Train.csv"
        elif difficulty == "challenge":
            if type == "dev":
                self.dataset_path = self.dataset_path + "ARC-Challenge/Arc-Challenge-Dev.csv"
            elif type == "test":
                self.dataset_path = self.dataset_path + "ARC-Challenge/Arc-Challenge-Test.csv"
            elif type == "train":
                self.dataset_path = self.dataset_path + "ARC-Challenge/Arc-Challenge-Train.csv"

        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.questions.append(row)



    def evaluate_n_questions(self, model, number_of_questions: int = 10):
        """Evaluate the model using the ARC dataset.

        Args:
        ----
        model: object
            The LLM model to be evaluated.
        number_of_questions: int, default=10
            The number of questions to evaluate.

        Returns:
        -------
        score: int
            The score of the model.
        total: int
            The total possible points.

        """
        score = 0
        total_possible_score = number_of_questions

        for question in self.questions[:number_of_questions]:
            print("\n")
            print("Question: ", question['question'])
            model_answer = model.predict(question['question'])
            normalized_model_answer = keep_only_alphanumeric(model_answer)
            print("Prediction: ", model_answer)
            print("Normalized Prediction: ", normalized_model_answer)
            correct_answer = question['AnswerKey']
            print("Correct Answer: ", correct_answer)


            if normalized_model_answer == correct_answer:
                score += 1

        return score, total_possible_score


class YourLLMModel:
    def predict(self, question):
        # Implement the prediction logic for your LLM model here.
        # For example, sending a request to a model API or using a local model.
        return "A"  # Dummy answer, replace with actual prediction logic.



if __name__ == "__main__":
    # Test benchmarking the model

    ## Load dataset
    arc_class = ArcBenhmark(dataset_path="data/raw/ARC/", arc_difficulty="easy", arc_type="dev")



    llm_model = YourLLMModel()
    score, max_score = arc_class.evaluate_n_questions(model=llm_model, number_of_questions=5)
    print(f"\nModel scored {score} out of 10 possible points")

