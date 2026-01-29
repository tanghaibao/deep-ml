import numpy as np

def verify_code_execution(
    test_cases: list[dict],
    numeric_tolerance: float = 1e-6
) -> dict:
    """
    Verify code execution results for a programming benchmark.
    
    Args:
        test_cases: List of dicts with keys:
            - 'expected': Expected output string
            - 'actual': Actual output string (or None if execution failed)
            - 'status': 'success', 'error', or 'timeout'
        numeric_tolerance: Tolerance for floating-point comparisons
        
    Returns:
        Dict with keys:
            - 'pass_rate': Proportion of passed tests (float, rounded to 4 decimals)
            - 'error_rate': Proportion of execution errors (float, rounded to 4 decimals)
            - 'passed_count': Number of passed tests (int)
            - 'total_count': Total number of tests (int)
            - 'verdicts': List of 'pass', 'fail', or 'error' for each test
    """
    # Your code here
    def normalize(s: str):
        if not s:
            