def extract_boxed_answer(response: str) -> str:
    """
    Extract the answer from within \boxed{...} in a model response.
    
    Args:
        response: The model's text response containing a boxed answer
    
    Returns:
        The content inside the last \boxed{}, or empty string if not found
    """
    # Your code here
    tag = r"\boxed{"
    if tag not in response:
        return ""
    part = response.rsplit(tag, 1)[-1]
    depth = 1
    extracted = ""
    for c in part:
        extracted += c
        if c == "{":
            depth += 1
        if c == "}":
            depth -= 1
        if depth == 0:
            break
    if extracted and extracted[-1] == "}":
        extracted = extracted[:-1]
    return extracted