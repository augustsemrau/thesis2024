"""Module containing the assessment of conversations using the AAS."""

# Local Imports
from thesis2024.utils import init_llm_langsmith
from thesis2024.models.AAS import AAS



def run_assessment(assessment_class, conversation):
    """Run the assessment."""
    assessment_output = assessment_class.predict(conversation)
    return assessment_output


if __name__ == "__main__":

    AAS_llm_model = init_llm_langsmith(llm_key=3, temp=0.5, langsmith_name="AAS TEST 1")
    AAS_class = AAS(llm_model=AAS_llm_model)

    assessment = run_assessment(assessment_class=AAS_class, conversation="This is a test conversation.")
    print(assessment)



