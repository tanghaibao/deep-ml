def calculate_data_quality_score(data: list, schema: dict) -> dict:
    """
    Calculate data quality metrics for ML pipeline monitoring.
    
    Args:
        data: list of dictionaries representing rows of data
        schema: dictionary defining expected columns and their types
                {'column_name': {'type': 'numeric'|'categorical'|'boolean', 'nullable': True|False}}
    
    Returns:
        dict with keys: 'completeness', 'type_validity', 'uniqueness_ratio', 'overall_score'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    def check_valid(v, t):
        if v is None:
            return t["nullable"]
        if t["type"] == "numeric":
            return isinstance(v, int) or isinstance(v, float)
        elif t["type"] == "categorical":
            return isinstance(v, str)
        elif t["type"] == "boolean":
            return isinstance(v, bool)
    
    if not data:
        return {}
    n = len(data)
    nulls = 0
    valid = 0
    