def analyze_ml_pipeline(tasks: list) -> dict:
    """
    Analyze an ML pipeline DAG for scheduling and critical path.
    """
    from collections import deque, defaultdict
    
    if not tasks:
        return {
            'execution_order': [],
            'earliest_start': {},
            'earliest_finish': {},
            'latest_start': {},
            'latest_finish': {},
            'slack': {},
            'critical_path': [],
            'makespan': 0
        }
    
    # Build task lookup and adjacency lists
    task_map = {t['id']: t for t in tasks}
    task_ids = list(task_map.keys())
    
    # Build graph structures
    in_degree = defaultdict(int)
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    
    for t in tasks:
        tid = t['id']
        for dep in t.get('dependencies', []):
            out_edges[dep].append(tid)
            in_edges[tid].append(dep)
            in_degree[tid] += 1
    
    # Topological sort using Kahn's algorithm with al