# Implement your function below.

def run_etl(csv_text: str) -> list[tuple[str, float]]:
    """Run a simple ETL pipeline over CSV text with header user_id,event_type,value.

    Returns a sorted list of (user_id, total_value) for event_type == "purchase".
    """
    from collections import defaultdict

    res = defaultdict(int)
    columns = {}
    for row in csv_text.splitlines():
        atoms = row.split(",")
        if "user_id" in atoms:
            for i, atom in enumerate(atoms):
                columns[atom] = i
        user_id = atoms[columns["user_id"]].strip()
        event_type = atoms[columns["event_type"]].strip()
        value = atoms[columns["value"]]
        if event_type != "purchase":
            continue
        try:
            value = float(value)
        except ValueError:
            continue
        res[user_id] += value
    return sorted(res.items())