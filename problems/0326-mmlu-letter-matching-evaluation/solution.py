import string
from collections import defaultdict

def mmlu_letter_matching(model_outputs: list[str], ground_truth: list[str], subjects: list[str]) -> dict:
    """
    Evaluate MMLU predictions using letter-matching.
    
    Args:
        model_outputs: List of model generated responses
        ground_truth: List of correct answer letters (A, B, C, or D)
        subjects: List of subject names for each question
    
    Returns:
        Dictionary with evaluation metrics
    """
    def normalize(ans):
        cleaned = "".join(x for x in ans if x in string.ascii_letters + " ")
        singles = [x.upper() for x in cleaned.split() if len(x) == 1]
        if not singles:
            return "invalid"
        ans = singles[-1]
        if ans in "ABCD":
            return ans
        return "invalid"
    
    valid = defaultdict(int) 
    correct = defaultdict(int) 
    total = defaultdict(int) 
    for output, truth, subject in zip(model_outputs, ground_truth, subjects):
        norm = 